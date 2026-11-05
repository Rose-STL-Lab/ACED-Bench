"""
orchestrator.py

Manages the interaction loop between the Scientist Agent and the World Model.

Responsibilities:
    - Present the initial problem (variables + question) to the Scientist
    - Relay queries from Scientist to World Model
    - Enforce query budget
    - Log all interactions
    - Collect and evaluate the final answer

Usage:
    orchestrator = Orchestrator(
        world_model=world,
        scientist=scientist,
        question=question,
        max_queries=10,
    )
    result = orchestrator.run()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from schemas import Question, QueryResult

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """Record of a single turn in the interaction."""
    turn_number: int
    turn_type: str  # "query", "answer", "system"
    scientist_input: str
    world_output: Optional[str]
    parsed_query: Optional[Dict[str, Any]]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # Enhanced logging fields
    raw_llm_response: Optional[str] = None
    reasoning: Optional[str] = None
    scientist_memory: Optional[str] = None
    data_summary: Optional[str] = None
    # Full multi-turn conversation (Active+Code and Modular agents)
    llm_transcript: Optional[List[Dict[str, Any]]] = None
    # Structured record of each code execution: [{"code": ..., "output": ...}, ...]
    code_rounds: Optional[List[Dict[str, str]]] = None
    # Per-turn phase summary (Modular agent)
    phase_summary: Optional[Dict[str, Any]] = None
    sample_usage_after: Optional[Dict[str, Any]] = None
    limit_event: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        # Order keys so the most informative fields (turn summary, action, reasoning)
        # come first and the bulky transcripts come last.
        d: Dict[str, Any] = {
            "turn_number": self.turn_number,
            "turn_type": self.turn_type,
            "timestamp": self.timestamp,
        }
        if self.phase_summary is not None:
            d["phase_summary"] = self.phase_summary
        if self.limit_event is not None:
            d["limit_event"] = self.limit_event
        d["scientist_input"] = self.scientist_input
        if self.reasoning is not None:
            d["reasoning"] = self.reasoning
        if self.scientist_memory is not None:
            d["scientist_memory"] = self.scientist_memory
        if self.parsed_query is not None:
            d["parsed_query"] = self.parsed_query
        if self.data_summary is not None:
            d["data_summary"] = self.data_summary
        if self.sample_usage_after is not None:
            d["sample_usage_after"] = self.sample_usage_after
        if self.world_output is not None:
            d["world_output"] = self.world_output
        if self.code_rounds is not None:
            d["code_rounds"] = self.code_rounds
        if self.raw_llm_response is not None:
            d["raw_llm_response"] = self.raw_llm_response
        if self.llm_transcript is not None:
            d["llm_transcript"] = self.llm_transcript
        return d


@dataclass
class ExperimentResult:
    """Final result of an experiment run."""
    question: Question
    scientist_answer: str
    ground_truth: Any
    is_correct: Optional[bool]   # None: not checked in-run (causal-decision questions)
    turns: List[Turn]
    total_queries: int
    max_queries: int
    final_scientist_memory: str = ""
    # Provenance — set from Orchestrator fields
    scientist_model: Optional[str] = None
    world_model_name: Optional[str] = None
    dataset_file: Optional[str] = None
    resource_limits: Optional[Dict[str, Any]] = None
    resource_usage: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        # Provenance at the top so it's immediately visible in the file
        if self.scientist_model is not None:
            d["scientist_model"] = self.scientist_model
        if self.world_model_name is not None:
            d["world_model"] = self.world_model_name
        if self.dataset_file is not None:
            d["dataset_file"] = self.dataset_file
        if self.resource_limits is not None:
            d["resource_limits"] = self.resource_limits
        if self.resource_usage is not None:
            d["resource_usage"] = self.resource_usage

        # ── High-level experiment summary (computed from turns) ───────────────
        turn_dicts = [t.to_dict() for t in self.turns]
        phase_summaries = [
            t.get("phase_summary") for t in turn_dicts if t.get("phase_summary")
        ]
        total_code_rounds = sum(p.get("code_rounds_this_turn", 0) for p in phase_summaries)
        total_code_errors = sum(p.get("code_errors_this_turn", 0) for p in phase_summaries)
        total_llm_calls = sum(p.get("n_llm_calls", 0) for p in phase_summaries)
        code_rounds_per_turn = [
            {
                "turn": p.get("turn_number"),
                "phases": p.get("phases_run"),
                "code_rounds": p.get("code_rounds_this_turn", 0),
                "code_errors": p.get("code_errors_this_turn", 0),
                "confidence_after": p.get("confidence_after_turn"),
                "action": p.get("action_type"),
            }
            for p in phase_summaries
        ]
        confidence_trajectory = [
            p.get("confidence_after_turn") for p in phase_summaries
            if p.get("confidence_after_turn") is not None
        ]
        final_confidence = confidence_trajectory[-1] if confidence_trajectory else None

        d["experiment_summary"] = {
            "outcome": {True: "correct", False: "incorrect"}.get(self.is_correct, "not_checked"),
            "total_turns": len(self.turns),
            "total_queries_used": self.total_queries,
            "max_queries": self.max_queries,
            "total_llm_calls": total_llm_calls,
            "total_code_rounds": total_code_rounds,
            "total_code_errors": total_code_errors,
            "final_confidence": final_confidence,
            "confidence_trajectory": confidence_trajectory,
            "per_turn": code_rounds_per_turn,
        }
        if self.resource_usage is not None:
            d["experiment_summary"]["resource_usage"] = self.resource_usage

        d.update({
            "question": self.question.to_dict(),
            "scientist_answer": self.scientist_answer,
            "ground_truth": self.ground_truth,
            "is_correct": self.is_correct,
            "total_queries": self.total_queries,
            "max_queries": self.max_queries,
            "final_scientist_memory": self.final_scientist_memory,
            "turns": turn_dicts,
        })
        return d
    
    def save(self, filepath: str) -> None:
        """Save result to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class Orchestrator:
    """
    Manages the interaction between Scientist Agent and World Model.
    
    The orchestrator:
        1. Presents the problem to the scientist
        2. Processes scientist's queries through the world model
        3. Tracks budget and history
        4. Collects the final answer
    """
    world_model: Any  # WorldModel instance
    scientist: Any    # ActiveAgent, ActiveCodeAgent or ModularAgent
    question: Question
    max_queries: int = 10
    log_dir: str = "./experiment_logs"
    # Provenance — logged into the experiment JSON for reproducibility
    scientist_model: Optional[str] = None
    world_model_name: Optional[str] = None
    dataset_file: Optional[str] = None
    limit_variant: str = "none"
    max_turns: Optional[int] = None
    max_total_samples: Optional[int] = None
    max_samples_per_query: Optional[int] = None
    sample_accounting: str = "rows"

    # Internal state
    _turns: List[Turn] = field(default_factory=list, init=False)
    _query_count: int = field(default=0, init=False)
    _experiment_id: str = field(default="", init=False)
    _final_turn_warning_sent: bool = field(default=False, init=False)
    _limit_stop_reason: Optional[str] = field(default=None, init=False)
    
    def __post_init__(self):
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self._experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    def run(self) -> ExperimentResult:
        """
        Run the full experiment loop.
        
        Returns:
            ExperimentResult with the outcome
        """
        logger.info(f"Starting experiment {self._experiment_id}")
        logger.info(f"Question: {self.question.question_text}")
        logger.info(f"Budget: {self.max_queries} queries")
        if hasattr(self.world_model, "reset_sample_usage"):
            self.world_model.reset_sample_usage()
        self._configure_world_model_limits()
        
        # Step 1: Initialize scientist with world info and question
        world_info = self.world_model.get_world_info()
        self.scientist.initialize(world_info, self.question, self.max_queries)
        self.scientist.receive_system_message(self._resource_limits_message())
        
        # Step 2: Main interaction loop
        forced_answer_retries = 0
        MAX_FORCED_RETRIES = 2  # after this many budget-exhausted prompts, give up
        failed_query_retries = 0
        MAX_FAILED_QUERY_RETRIES = 8  # validation/parser failures do not spend budget
        while True:
            if self._would_exceed_turn_limit():
                answer = self._force_limit_give_up("max_turns_exhausted")
                break
            self._warn_if_final_turn()

            # Get scientist's next action
            action = self.scientist.get_next_action()

            if action["type"] == "query":
                if self._is_final_allowed_turn():
                    self._limit_stop_reason = "final_turn_query_not_executed"
                    forced_action = dict(action)
                    forced_action["type"] = "give_up"
                    forced_action["content"] = (
                        "Final turn query was not executed. The scientist asked for "
                        f"data on turn {len(self._turns) + 1}/{self.max_turns}; "
                        "no later turn would be available to analyze it. "
                        f"Requested query: {action.get('content', '')}"
                    )
                    self._log_answer_turn(
                        forced_action,
                        limit_event="final_turn_query_not_executed",
                    )
                    answer = "GIVE_UP: " + forced_action["content"]
                    break

                # Process query
                if self._query_count >= self.max_queries:
                    # Budget exhausted - force answer
                    forced_answer_retries += 1
                    if forced_answer_retries > MAX_FORCED_RETRIES:
                        logger.error(
                            f"Scientist refused to answer after {forced_answer_retries} "
                            f"budget-exhausted prompts; converting last response into forced answer."
                        )
                        forced_action = dict(action)
                        forced_action["type"] = "answer"
                        # Prefer reasoning/memory if content is just a query stub
                        forced_action["content"] = (
                            action.get("content")
                            or action.get("reasoning")
                            or action.get("raw_response", "")[:500]
                            or "FORCED_ANSWER: scientist failed to comply"
                        )
                        self._log_answer_turn(
                            forced_action,
                            limit_event="max_queries_exhausted_forced_answer",
                        )
                        answer = forced_action["content"]
                        break
                    logger.warning(
                        f"Query budget exhausted, forcing answer "
                        f"(attempt {forced_answer_retries}/{MAX_FORCED_RETRIES})"
                    )
                    self._notify_budget_exhausted()
                    continue

                result = self._process_query(action)
                self.scientist.receive_result(result)
                if self.limit_variant == "samples":
                    self.scientist.receive_system_message(self._sample_usage_message())
                if result.success:
                    failed_query_retries = 0
                else:
                    failed_query_retries += 1
                    err = result.error_message or "unknown query failure"
                    logger.warning(
                        f"Failed query retry {failed_query_retries}/"
                        f"{MAX_FAILED_QUERY_RETRIES}: {err}"
                    )
                    self.scientist.receive_system_message(
                        "<query_failed>\n"
                        f"Your last query failed and returned no data: {err}\n"
                        "Design a different valid query. "
                        f"{self._sample_retry_instruction()}\n"
                        "</query_failed>"
                    )
                    if failed_query_retries >= MAX_FAILED_QUERY_RETRIES:
                        give_up_action = {
                            "type": "give_up",
                            "content": (
                                "Too many consecutive failed data queries "
                                f"({failed_query_retries}). Last error: {err}"
                            ),
                            "raw_response": action.get("raw_response"),
                            "reasoning": action.get("reasoning"),
                            "scientist_memory": action.get("scientist_memory"),
                            "llm_transcript": action.get("llm_transcript"),
                            "code_rounds": action.get("code_rounds"),
                            "phase_summary": action.get("phase_summary"),
                        }
                        self._log_answer_turn(
                            give_up_action,
                            limit_event="too_many_failed_queries",
                        )
                        answer = "GIVE_UP: " + give_up_action["content"]
                        logger.error(answer)
                        break

                # Update the turn's data_summary with the full statistical summary
                # computed by the scientist during receive_result
                if self.scientist._query_history:
                    latest = self.scientist._query_history[-1]
                    self._turns[-1].data_summary = latest.get("data_summary")

            elif action["type"] == "answer":
                # Log the final answer turn with reasoning/memory
                self._log_answer_turn(action)
                answer = action["content"]
                logger.info(f"Scientist answer: {answer}")
                break

            elif action["type"] == "give_up":
                self._log_answer_turn(action)
                answer = "GIVE_UP: " + action.get("content", "No reason given")
                logger.warning(f"Scientist gave up: {answer}")
                break

            else:
                logger.error(f"Unknown action type: {action['type']}")
                continue

        # Step 3: Evaluate answer
        is_correct = self._evaluate_answer(answer, self.question.ground_truth)

        # Get final scientist memory
        final_memory = getattr(self.scientist, '_scientist_memory', '')

        result = ExperimentResult(
            question=self.question,
            scientist_answer=answer,
            ground_truth=self.question.ground_truth,
            is_correct=is_correct,
            turns=self._turns,
            total_queries=self._query_count,
            max_queries=self.max_queries,
            final_scientist_memory=final_memory,
            scientist_model=self.scientist_model,
            world_model_name=self.world_model_name,
            dataset_file=self.dataset_file,
            resource_limits=self._resource_limits_dict(),
            resource_usage=self._resource_usage_dict(),
        )
        
        # Save logs
        log_path = Path(self.log_dir) / f"experiment_{self._experiment_id}.json"
        result.save(str(log_path))
        logger.info(f"Experiment complete. Correct: {is_correct}. Log: {log_path}")
        
        return result

    def _configure_world_model_limits(self) -> None:
        """Apply resolved sample limits to the world model if it supports them."""
        if hasattr(self.world_model, "max_total_samples"):
            self.world_model.max_total_samples = self.max_total_samples
        if hasattr(self.world_model, "max_samples_per_query"):
            self.world_model.max_samples_per_query = self.max_samples_per_query
        if hasattr(self.world_model, "sample_accounting"):
            self.world_model.sample_accounting = self.sample_accounting

    def _resource_limits_dict(self) -> Dict[str, Any]:
        return {
            "limit_variant": self.limit_variant,
            "max_turns": self.max_turns,
            "max_total_samples": self.max_total_samples,
            "max_samples_per_query": self.max_samples_per_query,
            "sample_accounting": self.sample_accounting,
        }

    def _resource_usage_dict(self) -> Dict[str, Any]:
        sample_usage = {}
        if hasattr(self.world_model, "get_sample_usage"):
            sample_usage = self.world_model.get_sample_usage()
        units = sample_usage.get("sample_units_used", 0)
        return {
            "turns_used": len(self._turns),
            "sample_rows_used": sample_usage.get("sample_rows_used", 0),
            "sample_cells_used": sample_usage.get("sample_cells_used", 0),
            "sample_units_used": units,
            "sample_accounting": sample_usage.get("sample_accounting", self.sample_accounting),
            "limit_stop_reason": self._limit_stop_reason,
        }

    def _resource_limits_message(self) -> str:
        lines = ["<resource_limits>"]
        if self.limit_variant == "rounds":
            lines.append(
                f"Round-limited run: you have at most {self.max_turns} outer "
                "scientist turns total. Query, failed query, final answer, and "
                "give_up each count as one turn."
            )
            lines.append(
                "On the final allowed turn you must answer or give up. A query "
                "on the final turn will not be executed because there is no "
                "later turn to analyze it."
            )
        elif self.limit_variant == "samples":
            lines.append(
                "Sample-limited run: query count is still available, but total "
                "data collection is budgeted separately."
            )
            lines.append(
                f"Total sample budget: {self.max_total_samples} "
                f"{self.sample_accounting} per question."
            )
            if self.max_samples_per_query is not None:
                lines.append(
                    f"Per-query sample budget: {self.max_samples_per_query} "
                    f"{self.sample_accounting}."
                )
            lines.append(
                "For multi-condition intervention requests, each condition consumes "
                "its own rows. Over-budget queries return no data."
            )
        else:
            lines.append("No extra round or total-sample ablation limit is active.")
        lines.append("</resource_limits>")
        return "\n".join(lines)

    def _would_exceed_turn_limit(self) -> bool:
        return self.max_turns is not None and len(self._turns) >= self.max_turns

    def _is_final_allowed_turn(self) -> bool:
        return self.max_turns is not None and len(self._turns) + 1 >= self.max_turns

    def _warn_if_final_turn(self) -> None:
        if (
            self.max_turns is not None
            and len(self._turns) + 1 == self.max_turns
            and not self._final_turn_warning_sent
        ):
            self._final_turn_warning_sent = True
            self.scientist.receive_system_message(
                "<final_turn>\n"
                f"This is turn {self.max_turns}/{self.max_turns}, the final "
                "allowed scientist turn. Do not request more data. Submit your "
                "final answer now, or give up if the evidence is insufficient.\n"
                "</final_turn>"
            )

    def _force_limit_give_up(self, reason: str) -> str:
        self._limit_stop_reason = reason
        action = {
            "type": "give_up",
            "content": f"Resource limit reached before a final answer: {reason}",
        }
        self._log_answer_turn(action, limit_event=reason)
        logger.warning(action["content"])
        return "GIVE_UP: " + action["content"]

    def _sample_retry_instruction(self) -> str:
        if not hasattr(self.world_model, "get_sample_usage"):
            return "Keep sample size <= 10000."
        usage = self.world_model.get_sample_usage()
        remaining = usage.get("sample_units_remaining")
        if self.max_total_samples is not None and remaining is not None:
            return (
                f"Remaining sample budget: {remaining} {self.sample_accounting}. "
                f"Ask for <= {remaining} {self.sample_accounting}, or answer from "
                "existing data."
            )
        if self.max_samples_per_query is not None:
            return (
                f"Keep each query <= {self.max_samples_per_query} "
                f"{self.sample_accounting}."
            )
        max_samples = getattr(self.world_model, "max_samples", 10000)
        return f"Keep sample size <= {max_samples}."

    def _sample_usage_message(self) -> str:
        if not hasattr(self.world_model, "get_sample_usage"):
            return "<sample_budget>Sample usage unavailable.</sample_budget>"
        usage = self.world_model.get_sample_usage()
        remaining = usage.get("sample_units_remaining")
        remaining_text = (
            "unlimited"
            if remaining is None
            else f"{remaining} {usage.get('sample_accounting', self.sample_accounting)}"
        )
        return (
            "<sample_budget>\n"
            f"Sample usage so far: {usage.get('sample_units_used', 0)} "
            f"{usage.get('sample_accounting', self.sample_accounting)}. "
            f"Remaining: {remaining_text}.\n"
            "Future queries that exceed the remaining sample budget will return no data.\n"
            "</sample_budget>"
        )
    
    def _process_query(self, action: Dict[str, Any]) -> QueryResult:
        """Process a query through the world model, logging enhanced metadata."""
        query = action["content"]

        result = self.world_model.process_query(query)

        # Only consume budget on successful queries so that parse errors,
        # validation errors, etc. don't penalise the scientist.
        if result.success:
            self._query_count += 1

        logger.info(f"Query {self._query_count}/{self.max_queries} (success={result.success}): {query[:100]}...")

        # Compute data summary string for logging (scientist will also compute it)
        data_summary = None
        if result.success and result.data_file:
            # The scientist's receive_result will compute the full summary;
            # we grab the scientist's most recent history entry after receive_result
            # is called. For now, store what we can from the result.
            data_summary = result.preview

        # Log the turn with enhanced metadata
        sample_usage = None
        if hasattr(self.world_model, "get_sample_usage"):
            sample_usage = self.world_model.get_sample_usage()

        turn = Turn(
            turn_number=len(self._turns) + 1,
            turn_type="query",
            scientist_input=query,
            world_output=result.to_xml(),
            parsed_query=result.query.to_dict() if result.success else None,
            raw_llm_response=action.get("raw_response"),
            reasoning=action.get("reasoning"),
            scientist_memory=action.get("scientist_memory"),
            data_summary=data_summary,
            llm_transcript=action.get("llm_transcript"),
            code_rounds=action.get("code_rounds"),
            phase_summary=action.get("phase_summary"),
            sample_usage_after=sample_usage,
        )
        self._turns.append(turn)

        return result

    def _log_answer_turn(
        self,
        action: Dict[str, Any],
        limit_event: Optional[str] = None,
    ) -> None:
        """Log an answer or give_up turn with enhanced metadata."""
        sample_usage = None
        if hasattr(self.world_model, "get_sample_usage"):
            sample_usage = self.world_model.get_sample_usage()
        turn = Turn(
            turn_number=len(self._turns) + 1,
            turn_type=action["type"],
            scientist_input=action["content"],
            world_output=None,
            parsed_query=None,
            raw_llm_response=action.get("raw_response"),
            reasoning=action.get("reasoning"),
            scientist_memory=action.get("scientist_memory"),
            llm_transcript=action.get("llm_transcript"),
            code_rounds=action.get("code_rounds"),
            phase_summary=action.get("phase_summary"),
            sample_usage_after=sample_usage,
            limit_event=limit_event,
        )
        self._turns.append(turn)

    def _notify_budget_exhausted(self) -> None:
        """Notify scientist that query budget is exhausted."""
        message = f"""<budget_exhausted>
You have used all {self.max_queries} queries. 
You must now provide your final answer based on the data you have collected.
</budget_exhausted>"""
        self.scientist.receive_system_message(message)
    
    def _evaluate_answer(self, answer: str, ground_truth: Any) -> Optional[bool]:
        """
        A quick in-run check of the answer, logged as is_correct. The reported scores come from evaluate_struct.py
        and evaluate_decision.py; causal-decision questions are not checked here (None).
        """
        qtype = getattr(self.question, "question_type", "") or ""
        if qtype.startswith("advanced_"):
            return None

        # Normalize answer
        answer_normalized = answer.strip().lower()

        # Handle different ground truth types
        if isinstance(ground_truth, bool):
            # Yes/No questions
            positive = ["yes", "true", "1", "correct"]
            negative = ["no", "false", "0", "incorrect"]
            
            answer_is_positive = any(p in answer_normalized for p in positive)
            answer_is_negative = any(n in answer_normalized for n in negative)
            
            if answer_is_positive and not answer_is_negative:
                return ground_truth is True
            elif answer_is_negative and not answer_is_positive:
                return ground_truth is False
            else:
                # Ambiguous
                return False
                
        elif isinstance(ground_truth, set):
            # Set questions (e.g., "list all ancestors")
            # Try to extract variable names from answer
            # Find words that match variable names
            all_vars = set(self.world_model.simulator.get_nodes())
            found_vars = set()
            for var in all_vars:
                if var.lower() in answer_normalized:
                    found_vars.add(var)
            
            return found_vars == ground_truth
            
        elif isinstance(ground_truth, list):
            # Convert to set for comparison
            return self._evaluate_answer(answer, set(ground_truth))
            
        elif isinstance(ground_truth, str):
            # String matching
            return ground_truth.lower() in answer_normalized
            
        else:
            # Fallback: string comparison
            return str(ground_truth).lower() in answer_normalized
