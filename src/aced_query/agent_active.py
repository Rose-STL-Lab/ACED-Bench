"""
agent_active.py

The Active agent: each turn the LLM reads the question, its memory, and statistical summaries of the data it
has collected, then either requests more observational or interventional data or answers.

Usage:
    from agent_active import ActiveAgent
    from bedrock_llm import BedrockLLM

    scientist = ActiveAgent(llm=BedrockLLM(model_id="us.anthropic.claude-opus-4-6-v1"))
    scientist.initialize(world_info, question, max_queries=10)
    action = scientist.get_next_action()  # {"type": "query" | "answer" | "give_up", "content": "..."}
    scientist.receive_result(result)
"""

from __future__ import annotations

import itertools
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from schemas import Question, QueryResult, WorldInfo

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Scientist Agent
# -----------------------------------------------------------------------------

@dataclass
class ActiveAgent:
    """
    LLM-powered agent for causal discovery.

    The scientist must answer questions about causal relationships between
    variables in an unknown causal graph. It can collect:
        - Observational data: reveals correlations (but not causal direction)
        - Interventional data: do(X=x) severs X's incoming causal links and
          reveals which variables X causally affects

    Each turn, it decides whether to:
        - Request more data (observational or interventional)
        - Submit a final answer
    """
    llm: Any  # any object with generate(system_prompt, user_prompt)

    # State (set during initialize)
    world_info: Optional[WorldInfo] = field(default=None, init=False)
    question: Optional[Question] = field(default=None, init=False)
    max_queries: int = field(default=10, init=False)

    # History
    _query_history: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _queries_made: int = field(default=0, init=False)
    _system_messages: List[str] = field(default_factory=list, init=False)

    # Scientist memory: evolving causal understanding of the world
    _scientist_memory: str = field(default="", init=False)

    def initialize(
        self,
        world_info: WorldInfo,
        question: Question,
        max_queries: int,
    ) -> None:
        """
        Initialize the scientist with the causal discovery problem.

        Args:
            world_info: Variable descriptions and story (no graph structure)
            question: The causal question to answer
            max_queries: Maximum number of data queries allowed
        """
        self.world_info = world_info
        self.question = question
        self.max_queries = max_queries
        self._query_history = []
        self._queries_made = 0
        self._system_messages = []
        self._scientist_memory = ""

        logger.info(f"Scientist initialized. Question: {question.question_text}")

    def get_next_action(self) -> Dict[str, Any]:
        """
        Decide the next action: query for data or submit a causal answer.

        Returns:
            Dict with:
                - type: "query" | "answer" | "give_up"
                - content: the query string or answer
                - raw_response: the full LLM response
                - reasoning: extracted reasoning block
                - scientist_memory: current memory after this turn
        """
        if self.world_info is None or self.question is None:
            raise RuntimeError("Scientist not initialized. Call initialize() first.")

        system_prompt = self._SYSTEM_PROMPT
        user_prompt = self._get_decision_user_prompt()

        response = self.llm.generate(system_prompt, user_prompt)
        logger.debug(f"Scientist raw response:\n{response}")

        reasoning = self._extract_reasoning(response)
        if reasoning:
            logger.info(f"Scientist reasoning: {reasoning[:300]}...")

        self._update_memory_from_response(response)

        action = self._parse_action(response)

        # Truncation recovery: if the response has content but no parseable
        # action tag (model was too verbose and got cut off), ask it to emit
        # just the action tag in a short follow-up call.
        if (action["type"] == "give_up"
                and "Parsing failed" in action.get("content", "")
                and response.strip()):
            logger.warning("Response appears truncated — requesting action completion.")
            completion = self.llm.generate(
                "You are a discovery scientist. Output ONLY the action tag, nothing else.",
                f"Your previous response was cut off before the <action> tag. "
                f"Based on your reasoning below, output ONLY the appropriate "
                f"action tag now.\n\n{response[-2000:]}\n\n"
                f"Reply with EXACTLY one line:\n"
                f'<action type="query|answer|give_up">your query or answer</action>',
            )
            retry_action = self._parse_action(completion)
            if retry_action["type"] != "give_up":
                logger.info("Truncation recovery succeeded: %s", retry_action["type"])
                action = retry_action
                response = response + "\n[COMPLETION]\n" + completion

        action["raw_response"] = response
        action["reasoning"] = reasoning
        action["scientist_memory"] = self._scientist_memory

        logger.info(f"Scientist action: type={action['type']}, content={action['content'][:100]}...")

        return action

    def receive_result(self, result: QueryResult) -> None:
        """
        Receive the result of a data query.

        Args:
            result: QueryResult from the world model
        """
        # Only count successful queries toward budget (aligned with orchestrator)
        if result.success:
            self._queries_made += 1

        data_summary = ""
        if result.success and result.data_file:
            data_summary = self._compute_data_summary(result.data_file, result.query.interventions)
            logger.info(f"Statistical summary for query:\n{data_summary}")

        self._query_history.append({
            "query": result.query.raw_query,
            "success": result.success,
            "result_xml": result.to_xml(),
            "data_file": result.data_file,
            "n_rows": result.n_rows,
            "interventions": result.query.interventions,
            "data_summary": data_summary,
            "scientist_memory_snapshot": self._scientist_memory,
        })

        logger.info(f"Scientist received result: success={result.success}, n={result.n_rows}")

    def _compute_data_summary(
        self,
        data_file: str,
        interventions: List[Dict[str, str]],
        max_vars: int = 20,
    ) -> str:
        """
        Compute a comprehensive statistical summary of the data.

        Includes marginal distributions, pairwise independence tests (chi-squared
        with Cramer's V), conditional distributions for top associated pairs,
        and interventional shift analysis when applicable.
        """
        try:
            df = pd.read_csv(data_file)
        except Exception as e:
            logger.warning(f"Could not read data file {data_file}: {e}")
            return "(Could not read data)"

        if df.empty:
            return "(No data)"

        n = len(df)
        sections = []

        sections.append(f"=== DATA SUMMARY (N={n}, {len(df.columns)} variables) ===")

        if interventions:
            cond_strs = ["do(" + ", ".join(f"{k}={v}" for k, v in c.items()) + ")" for c in interventions]
            sections.append(f"Intervention(s): {' | '.join(cond_strs)}")

        # For marginal distributions, drop the indicator column if present
        data_cols = [c for c in df.columns if c != "__intervention__"]
        marginal_lines = ["MARGINAL DISTRIBUTIONS:"]
        for col in data_cols[:max_vars]:
            vc = df[col].value_counts(normalize=True).sort_index()
            counts_str = ", ".join(f"{state}: {pct:.0%}" for state, pct in vc.items())
            marginal_lines.append(f"  {col}: {counts_str}")
        if len(data_cols) > max_vars:
            marginal_lines.append(f"  ... and {len(data_cols) - max_vars} more variables")
        sections.append("\n".join(marginal_lines))

        stats_df = df[data_cols]
        pairwise_stats = self._compute_pairwise_stats(stats_df)
        if pairwise_stats:
            pair_lines = ["PAIRWISE ASSOCIATION TESTS (sorted by significance):"]
            for var1, var2, chi2, p, v in pairwise_stats:
                sig = ""
                if p < 0.001:
                    sig = " ***"
                elif p < 0.01:
                    sig = " **"
                elif p < 0.05:
                    sig = " *"
                pair_lines.append(f"  {var1} vs {var2}: chi2={chi2:.1f}, p={p:.4f}, V={v:.3f}{sig}")
            pair_lines.append("  (* p<0.05, ** p<0.01, *** p<0.001)")
            sections.append("\n".join(pair_lines))

        if pairwise_stats:
            cond_str = self._compute_conditional_distributions(stats_df, pairwise_stats)
            if cond_str:
                sections.append("CONDITIONAL DISTRIBUTIONS (top associated pairs):\n" + cond_str)

        if interventions:
            if len(interventions) == 1:
                shift_str = self._compute_interventional_shift(df, interventions[0])
            else:
                shift_str = self._compute_cross_condition_shift(df, interventions)
            if shift_str:
                sections.append("CAUSAL EFFECT ANALYSIS:\n" + shift_str)

        return "\n\n".join(sections)

    def receive_system_message(self, message: str) -> None:
        """Receive a system message (e.g., budget warning)."""
        self._system_messages.append(message)
        logger.info(f"Scientist received system message: {message[:100]}...")

    def _strip_think_block(self, response: str) -> str:
        """Remove Qwen-style <think>...</think> chain-of-thought block."""
        stripped = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        return stripped.strip()

    def _update_memory_from_response(self, response: str) -> None:
        """Extract and update scientist memory from LLM response."""
        response = self._strip_think_block(response)
        pattern = r'<scientist_memory>(.*?)</scientist_memory>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            self._scientist_memory = match.group(1).strip()
            logger.info(f"Scientist memory updated: {self._scientist_memory[:200]}...")
        else:
            logger.warning("No <scientist_memory> block found in LLM response")

    def _extract_reasoning(self, response: str) -> str:
        """Extract the reasoning block from LLM response."""
        response = self._strip_think_block(response)
        pattern = r'<reasoning>(.*?)</reasoning>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    # -------------------------------------------------------------------------
    # Statistical Analysis Helpers
    # -------------------------------------------------------------------------

    def _compute_pairwise_stats(
        self,
        df: pd.DataFrame,
        max_pairs: int = 15,
    ) -> List[Tuple[str, str, float, float, float]]:
        """
        Compute chi-squared independence tests and Cramer's V for all variable pairs.

        Returns list of (var1, var2, chi2, p_value, cramers_v) sorted by p-value ascending.
        """
        cols = list(df.columns)
        results = []
        for var1, var2 in itertools.combinations(cols, 2):
            try:
                contingency = pd.crosstab(df[var1], df[var2])
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue
                chi2, p, dof, expected = chi2_contingency(contingency, correction=True)
                n = len(df)
                min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
                cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0
                results.append((var1, var2, chi2, p, cramers_v))
            except Exception:
                continue
        results.sort(key=lambda x: x[3])
        return results[:max_pairs]

    def _compute_conditional_distributions(
        self,
        df: pd.DataFrame,
        pairs: List[Tuple[str, str, float, float, float]],
        top_k: int = 5,
    ) -> str:
        """Compute P(Y|X=x) for top associated pairs."""
        lines = []
        for var1, var2, _, _, _ in pairs[:top_k]:
            n_states_1 = df[var1].nunique()
            n_states_2 = df[var2].nunique()
            if n_states_1 <= n_states_2:
                cond_var, target_var = var1, var2
            else:
                cond_var, target_var = var2, var1

            try:
                ct = pd.crosstab(df[cond_var], df[target_var], normalize='index')
                lines.append(f"  P({target_var} | {cond_var}):")
                for state in ct.index:
                    dist_parts = [f"{col}={pct:.0%}" for col, pct in ct.loc[state].items()]
                    lines.append(f"    {cond_var}={state}: {', '.join(dist_parts)}")
            except Exception:
                continue
        return "\n".join(lines)

    def _compute_interventional_shift(
        self,
        df: pd.DataFrame,
        interventions,
    ) -> str:
        """
        Compare current interventional data with the most recent observational baseline.

        Returns a string describing which variables shifted under the intervention.
        A significant shift in variable Y after do(X=x) is evidence that X causally
        affects Y. No shift means X does not causally affect Y.
        """
        # Flatten list-of-dicts to single dict
        if isinstance(interventions, list):
            merged = {}
            for d in interventions:
                if isinstance(d, dict):
                    merged.update(d)
            interventions = merged

        obs_file = None
        for h in reversed(self._query_history):
            if h["success"] and not h.get("interventions"):
                obs_file = h.get("data_file")
                break

        if not obs_file:
            return "(No observational baseline available for comparison)"

        try:
            obs_df = pd.read_csv(obs_file)
        except Exception:
            return "(Could not read observational baseline)"

        lines = []
        interv_str = ", ".join(f"{k}={v}" for k, v in interventions.items())
        lines.append(f"  Intervention: do({interv_str})")
        lines.append("  Variable shifts (interventional vs observational):")
        lines.append("  [Significant shift = X likely has a causal effect on this variable]")
        lines.append("  [No shift = X likely does NOT causally affect this variable]")

        intervened_vars = set(interventions.keys())
        for col in df.columns:
            if col in intervened_vars:
                continue
            if col not in obs_df.columns:
                continue
            try:
                int_dist = df[col].value_counts(normalize=True).sort_index()
                obs_dist = obs_df[col].value_counts(normalize=True).sort_index()
                all_states = sorted(set(int_dist.index) | set(obs_dist.index))
                shifts = []
                for state in all_states:
                    obs_pct = obs_dist.get(state, 0.0)
                    int_pct = int_dist.get(state, 0.0)
                    delta = int_pct - obs_pct
                    if abs(delta) >= 0.02:
                        shifts.append(f"{state}: {obs_pct:.0%}->{int_pct:.0%} (delta={delta:+.0%})")
                if shifts:
                    lines.append(f"    {col}: SHIFTED — {'; '.join(shifts)}")
                else:
                    lines.append(f"    {col}: no significant change")
            except Exception:
                continue

        return "\n".join(lines)

    def _compute_cross_condition_shift(
        self,
        df: pd.DataFrame,
        interventions: List[Dict[str, str]],
    ) -> str:
        """
        Compare variable distributions across multiple intervention conditions.

        For multi-condition queries (e.g., do(X=a) vs do(X=b)), groups the
        stacked DataFrame by the __intervention__ label and shows how each
        non-intervened variable's distribution differs across conditions.
        """
        if "__intervention__" not in df.columns:
            return ""

        all_intervened = set()
        for c in interventions:
            all_intervened.update(c.keys())

        condition_labels = df["__intervention__"].unique().tolist()
        groups = {lbl: df[df["__intervention__"] == lbl] for lbl in condition_labels}

        lines = ["  Cross-condition comparison:"]
        lines.append(f"  [Significant shift across conditions = the intervened variable causally affects this variable]")

        data_cols = [c for c in df.columns if c != "__intervention__" and c not in all_intervened]
        for col in data_cols:
            try:
                dists = {}
                for lbl, grp in groups.items():
                    dists[lbl] = grp[col].value_counts(normalize=True).sort_index()
                all_states = sorted(set().union(*[d.index for d in dists.values()]))
                # Compute max delta across any pair of conditions
                max_delta = 0.0
                for state in all_states:
                    vals = [dists[lbl].get(state, 0.0) for lbl in condition_labels]
                    max_delta = max(max_delta, max(vals) - min(vals))
                if max_delta >= 0.05:
                    state_strs = []
                    for state in all_states:
                        per_cond = " vs ".join(f"{dists[lbl].get(state, 0.0):.0%}" for lbl in condition_labels)
                        state_strs.append(f"{state}: {per_cond}")
                    lines.append(f"    {col}: SHIFTED — {'; '.join(state_strs)}")
                else:
                    lines.append(f"    {col}: no significant change across conditions")
            except Exception:
                continue

        cond_header = " | ".join(condition_labels)
        lines.insert(2, f"  Conditions: {cond_header}")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Prompt Construction
    # -------------------------------------------------------------------------

    _SYSTEM_PROMPT = """You are a discovery scientist. Your goal is to answer the question you are given by collecting and analyzing \
data from an unknown system. Read the question carefully and decide for yourself what kind of answer it is asking for and what \
approach will get you there.

AVAILABLE ACTIONS (choose one each turn):
1. QUERY - Request data. Choose a sample size that is sufficient for the analysis you plan to run, but not wasteful. Specify the kind of data:
   - Observational: "Give me N observational samples of variables A, B, C"
     → samples drawn from the system's natural data-generating process; reveals
       how variables jointly behave (associations, correlations).
   - Interventional: "Give me N samples of A, B, and X where we intervene to set X=value"
     → samples where X is fixed to the given value, severing X's incoming edges;
       reveals the downstream effect of fixing X.
2. ANSWER - Submit your final answer when confident. Match the form the question implies (a Yes/No, a single variable, a list, a \
label, etc.). Variable names must match the VARIABLES catalog exactly.
3. GIVE_UP - If the question cannot be answered with available resources.

HOW TO THINK:

Think like a scientist running experiments. Different questions call for different
approaches — choose the data type and analysis that actually answers what is being
asked, rather than defaulting to one recipe.

Two general principles to keep in mind:
  - Correlation in observational data is not the same as a causal effect.
  - Interventional data reveals what happens when you fix a variable, which is
    a different question from how variables co-vary in the wild.

Strategy:
1. Restate what the question is asking and what kind of evidence would settle it.
2. Choose data (observational / interventional) and a sample size accordingly.
3. Focus only on variables relevant to the question.
4. Prefer the smallest experiment that can answer the question.

Before answering, ensure:
- You actually collected the kind of data that can answer this question.
- Your conclusion is grounded in observed evidence, not assumption.

OUTPUT FORMAT (required, in this exact order):
<reasoning>[Analysis, updated understanding, hypothesis, decision rationale]</reasoning>
<action type="query|answer|give_up">[Query, answer, or reason]</action>
<scientist_memory>
Tested:
Known:
Uncertain:
Next:
</scientist_memory>
Do NOT copy the example text — write your actual analysis and actual query/answer."""

    def _get_decision_user_prompt(self) -> str:
        """User prompt with current state — inverted pyramid structure."""
        # =======================================================================
        # SECTION 1: GOAL (most important)
        # =======================================================================
        goal_section = f"""════════════════════════════════════════════════════════════════════════════════
YOUR QUESTION: {self.question.question_text}
════════════════════════════════════════════════════════════════════════════════"""

        # =======================================================================
        # SECTION 2: CONSTRAINTS
        # =======================================================================
        remaining = self.max_queries - self._queries_made
        if remaining <= 0:
            budget_str = f">> BUDGET: {self._queries_made}/{self.max_queries} queries used — NO QUERIES LEFT — You MUST answer now!"
        elif remaining <= 2:
            budget_str = f">> BUDGET: {self._queries_made}/{self.max_queries} queries used — Only {remaining} remaining — Answer soon!"
        else:
            budget_str = f"BUDGET: {self._queries_made}/{self.max_queries} queries used — {remaining} remaining"

        non_intervenable = self.world_info.non_intervenable_variables
        if non_intervenable:
            interv_lines = ["INTERVENTION LIMITS — Cannot intervene on (non-manipulable variables):"]
            for var, reason in non_intervenable.items():
                interv_lines.append(f"  - {var}: {reason}")
            interv_str = "\n".join(interv_lines)
        else:
            interv_str = "INTERVENTION LIMITS: All variables are intervenable"

        constraints_section = f"""{budget_str}
{interv_str}"""

        if self._system_messages:
            constraints_section += "\n>> SYSTEM ALERTS:\n" + "\n".join(f"  - {msg}" for msg in self._system_messages)

        # =======================================================================
        # SECTION 3: LATEST QUERY RESULT
        # =======================================================================
        if self._query_history:
            latest = self._query_history[-1]
            status = "SUCCESS" if latest["success"] else "FAILED"
            latest_lines = [f"─── LATEST QUERY RESULT (Query {len(self._query_history)}) ───"]
            latest_lines.append(f"Status: {status}")
            latest_lines.append(f"Request: {latest['query']}")
            if latest["success"]:
                latest_lines.append(f"Samples: {latest['n_rows']}")
                latest_lines.append("─" * 60)
                if latest.get("data_summary"):
                    latest_lines.append(latest["data_summary"])
            else:
                error_msg = latest.get("result_xml", "(no details)")
                latest_lines.append(f"Error: Query failed — {error_msg}")
                latest_lines.append("─" * 60)
            latest_section = "\n".join(latest_lines)
        else:
            latest_section = (
                "─── NO QUERIES YET ───\n"
            )

        # =======================================================================
        # SECTION 4: YOUR CAUSAL MEMORY
        # =======================================================================
        if self._scientist_memory:
            memory_section = f"""YOUR CURRENT CAUSAL UNDERSTANDING (from previous turn):
{self._scientist_memory}"""
        else:
            memory_section = """YOUR CURRENT CAUSAL UNDERSTANDING:
(Empty — this is your first turn. After analyzing data, update your memory to track your evolving causal map: which variables are \
associated, which causal directions are confirmed, what remains uncertain.)"""

        # =======================================================================
        # SECTION 5: PAST QUERIES SUMMARY
        # =======================================================================
        if len(self._query_history) > 1:
            history_lines = ["PAST QUERIES SUMMARY:"]
            for i, h in enumerate(self._query_history[:-1], 1):
                status_icon = "[OK]" if h["success"] else "[FAIL]"
                if h["success"]:
                    conditions = h.get("interventions") or []
                    if conditions:
                        parts = ["do(" + ", ".join(f"{k}={v}" for k, v in c.items()) + ")" for c in conditions]
                        interv = " " + " | ".join(parts)
                    else:
                        interv = " observational"
                    history_lines.append(f"  Query {i}: {status_icon} {h['n_rows']} samples —{interv}")
                else:
                    history_lines.append(f"  Query {i}: {status_icon} Failed")
            history_lines.append(f"  Query {len(self._query_history)}: (Latest — see full results above)")
            history_section = "\n".join(history_lines)
        else:
            history_section = ""

        # =======================================================================
        # SECTION 6: VARIABLES (always shown in full — needed for valid intervention queries)
        # =======================================================================
        if self._queries_made == 0:
            variables_section = f"""AVAILABLE VARIABLES (use exact names and state values for interventions):
{self.world_info.get_variable_catalog()}

CONTEXT: {self.world_info.story}"""
        else:
            variables_section = f"""AVAILABLE VARIABLES (use exact names and state values for interventions):
{self.world_info.get_variable_catalog()}"""

        # =======================================================================
        # ASSEMBLE FINAL PROMPT
        # =======================================================================
        sections = [
            goal_section,
            "",
            variables_section,
            "",
            constraints_section,
            "",
            latest_section,
            "",
            memory_section,
        ]

        if history_section:
            sections.extend(["", history_section])

        sections.extend([
            "",
            "=" * 80,
            "Now: Analyze the latest data → Update your memory → Decide your next action.",
            "Output: <reasoning>, <action>, and <scientist_memory> blocks (in that order)."
        ])

        return "\n".join(sections)

    # -------------------------------------------------------------------------
    # Response Parsing
    # -------------------------------------------------------------------------

    def _parse_action(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into an action."""
        response = self._strip_think_block(response)
        pattern = r'<action\s+type="(\w+)">\s*(.*?)\s*</action>'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)

        if match:
            action_type = match.group(1).lower()
            content = match.group(2).strip()

            if "your natural language query here" in content.lower() or "your answer here" in content.lower():
                logger.warning("LLM output example text instead of actual content, falling through to recovery")
            elif action_type in ("query", "answer", "give_up"):
                return {"type": action_type, "content": content}

        # Fallback: try to infer intent
        response_lower = response.lower()

        if any(phrase in response_lower for phrase in ["my answer is", "the answer is", "i conclude"]):
            return {"type": "answer", "content": self._extract_answer(response)}

        query = self._extract_query(response)
        if query:
            return {"type": "query", "content": query}

        logger.warning(f"Could not parse action from response: {response[:200]}...")
        return {"type": "give_up", "content": f"Parsing failed: {response[:100]}"}

    def _extract_answer(self, response: str) -> str:
        """Try to extract an answer from unstructured response."""
        patterns = [
            r"(?:my answer is|the answer is|i conclude)[:\s]+(.+?)(?:\.|$)",
            r"(?:yes|no)[,\s]*(.+)?",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(0).strip()

        sentences = response.split(".")
        return sentences[-1].strip() if sentences else response[:100]

    def _extract_query(self, response: str) -> Optional[str]:
        """Try to extract a query from unstructured response."""
        patterns = [
            r"(?:give me|show me|get|request|sample)[:\s]+(.+?)(?:\.|$)",
            r"(?:observe|intervention|do\()[:\s]*(.+?)(?:\.|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(0).strip()

        return None
