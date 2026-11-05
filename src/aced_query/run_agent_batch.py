"""
run_agent_batch.py

Run one agent (Active, Active+Code or Modular) on every question of every world in a directory.

The world model (the request parser) runs on --world-model; the agent runs on --scientist-backend /
--scientist-model. Score the output with evaluate_struct.py (structural questions) or evaluate_decision.py
(causal-decision questions).

Usage:
    python run_agent_batch.py --worlds-dir ../../data/query/core_struct --agent-type modular \
        --scientist-backend bedrock --scientist-model us.anthropic.claude-opus-4-6-v1 \
        --limit-variant rounds --max-turns 4 --run-name opus46_modular
"""

from __future__ import annotations

import argparse
import json
import logging
import glob
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_active import ActiveAgent
from agent_active_code import ActiveCodeAgent
from agent_modular import ModularAgent
from bn_simulator import BNSimulator
from orchestrator import Orchestrator
from schemas import Question
from json_converter import JSONToBIFConverter
from world_model import WorldModel

logger = logging.getLogger(__name__)


# The request parser used for every number in the paper.
PARSER_MODEL = "us.anthropic.claude-opus-4-6-v1"
AGENTS = {"active": ActiveAgent, "active_code": ActiveCodeAgent, "modular": ModularAgent}


def resolve_resource_limits(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Dict[str, Any]:
    """The resource limit: a turn budget (rounds; the paper's main protocol uses 4 turns) or a total sample budget
    (samples; the paper's ablation uses 5,000 rows)."""
    if args.limit_variant == "rounds":
        if args.max_total_samples is not None or args.max_samples_per_query is not None:
            parser.error("--limit-variant rounds only accepts --max-turns; use --limit-variant samples for sample budgets")
        return {"limit_variant": "rounds", "max_turns": args.max_turns, "max_total_samples": None,
                "max_samples_per_query": None, "sample_accounting": args.sample_accounting}
    if args.max_total_samples is None:
        parser.error("--limit-variant samples requires --max-total-samples")
    return {"limit_variant": "samples", "max_turns": None, "max_total_samples": args.max_total_samples,
            "max_samples_per_query": args.max_samples_per_query, "sample_accounting": args.sample_accounting}


def resource_limits_compatible(expected: Dict[str, Any], actual: Optional[Dict[str, Any]]) -> bool:
    """Return whether an existing per-experiment log matches this run's limits."""
    if actual is None:
        return False
    keys = [
        "limit_variant",
        "max_turns",
        "max_total_samples",
        "max_samples_per_query",
        "sample_accounting",
    ]
    return all(actual.get(k) == expected.get(k) for k in keys)


# ---------------------------------------------------------------------------
# World loading
# ---------------------------------------------------------------------------

def load_worlds(
    worlds_dir: Optional[str] = None,
    world_json: Optional[str] = None,
) -> List:
    """Load world JSON files. Returns list of (filepath, parsed_json)."""
    results = []
    if world_json:
        paths = [world_json]
    elif worlds_dir:
        paths = sorted(glob.glob(os.path.join(worlds_dir, "world_*.json")))
    else:
        raise ValueError("Provide --worlds-dir or --world-json")

    for p in paths:
        try:
            with open(p) as f:
                data = json.load(f)
            if data.get("questions"):
                results.append((p, data))
            else:
                logger.warning(f"Skipping {p}: no questions found")
        except Exception as e:
            logger.warning(f"Skipping {p}: {e}")
    return results


# ---------------------------------------------------------------------------
# Per-world setup
# ---------------------------------------------------------------------------

def setup_world_model(filepath: str, llm, output_dir: str, world_model_cls):
    """Create a BNSimulator + WorldModel from a world JSON file.

    The *llm* is any object with ``generate(system_prompt, user_prompt, max_new_tokens=None)``.
    """
    converter = JSONToBIFConverter(filepath)
    errors = converter.validate()
    if errors:
        raise ValueError(f"Validation errors: {errors}")

    # the network file (which holds the full model) lives in a private folder only while it is read
    with tempfile.TemporaryDirectory(prefix="aced_bn_") as d:
        bif = os.path.join(d, "world.bif")
        converter.convert(bif)
        simulator = BNSimulator.from_bif(bif)
    config = converter.get_world_config()

    world_model = world_model_cls(simulator=simulator, llm=llm, output_dir=output_dir)
    world_model.set_variable_descriptions(config["variable_descriptions"])
    if config.get("non_intervenable_variables"):
        world_model.set_non_intervenable_variables(config["non_intervenable_variables"])
    if config.get("story"):
        world_model.set_story(config["story"])

    return world_model, config


def load_done_experiments(log_dir: str) -> Dict:
    """Scan log_dir for experiment_*.json files. Return dict keyed by
    (basename(dataset_file), question_id) -> (log_path, parsed log dict).

    Used to resume a partially-completed batch by skipping questions
    whose per-experiment log already exists.
    """
    done: Dict = {}
    if not os.path.isdir(log_dir):
        return done
    for log_path in sorted(glob.glob(os.path.join(log_dir, "experiment_*.json"))):
        try:
            with open(log_path) as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not parse {log_path}: {e}")
            continue
        ds = data.get("dataset_file")
        qid = data.get("question", {}).get("metadata", {}).get("id")
        if ds is None or qid is None:
            continue
        key = (os.path.basename(ds), qid)
        # Keep the most recent log if duplicates exist (sorted ascending → last wins)
        done[key] = (log_path, data)
    return done


def result_from_prior_log(log_data: Dict, log_path: str, filepath: str,
                          world_name: str, topic: str, n_nodes: int,
                          topology: str,
                          question: Question) -> Dict[str, Any]:
    """Re-create the aggregated-results row from a previously saved experiment log."""
    turns = log_data.get("turns", []) or []
    last_turn = turns[-1] if turns else {}
    return {
        "world_file": filepath,
        "world_name": world_name,
        "topic": topic,
        "n_nodes": n_nodes,
        "topology": topology,
        "question_id": question.metadata.get("id", 0),
        "question_text": question.question_text,
        "question_type": question.question_type,
        "difficulty": question.metadata.get("difficulty", ""),
        "ground_truth": question.ground_truth,
        "raw_response": last_turn.get("raw_llm_response", ""),
        "extracted_answer": log_data.get("scientist_answer", ""),
        "total_queries": log_data.get("total_queries", 0),
        "max_queries": log_data.get("max_queries", 0),
        "resource_limits": log_data.get("resource_limits"),
        "resource_usage": log_data.get("resource_usage"),
        "resumed_from": log_path,
    }


def questions_from_config(config: dict) -> List[Question]:
    """Convert question dicts from a world config into Question objects."""
    questions = []
    for q in config.get("questions", []):
        questions.append(Question(
            question_type=q["question_type"],
            question_text=q["question_text"],
            ground_truth=q["ground_truth"],
            metadata={k: v for k, v in q.items()
                      if k not in ("question_type", "question_text", "ground_truth")},
        ))
    return questions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Run an agent on every question of every world in a directory",
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("--worlds-dir", default=None,
                        help="Directory containing world_*.json files")
    parser.add_argument("--world-json", default=None,
                        help="Path to a single world JSON file")

    # World model (request parser)
    parser.add_argument("--world-model", default=PARSER_MODEL,
                        help=f"Model id of the request parser (default: {PARSER_MODEL}, the paper's parser)")
    parser.add_argument("--world-model-backend", choices=["bedrock", "openai"], default="bedrock",
                        help="Backend of the request parser")
    parser.add_argument("--world-model-base-url", default=None,
                        help="API base URL for an openai-backend parser (falls back to OPENAI_BASE_URL)")
    parser.add_argument("--world-model-api-key", default=None,
                        help="API key for an openai-backend parser (falls back to OPENAI_API_KEY)")

    # Agent (scientist) LLM
    parser.add_argument("--scientist-backend", choices=["bedrock", "openai", "gemini"], default="bedrock",
                        help="Agent backend: AWS Bedrock, an OpenAI-compatible API (OpenAI, vLLM, ...), or Gemini")
    parser.add_argument("--scientist-model", required=True, help="Agent model id for that backend")
    parser.add_argument("--scientist-base-url", default=None,
                        help="API base URL (openai backend), e.g. http://localhost:8000/v1 for vLLM. "
                             "Falls back to OPENAI_BASE_URL")
    parser.add_argument("--scientist-api-key", default=None,
                        help="API key (openai backend). Falls back to OPENAI_API_KEY")

    parser.add_argument("--agent-type", choices=sorted(AGENTS), default="active",
                        help="active = Active, active_code = Active+Code, modular = Modular (INIT/CODE/ANALYSIS/DESIGN)")
    parser.add_argument("--max-queries", "-n", type=int, default=10,
                        help="Cap on data queries per question (with the 4-turn budget, the turn limit binds first)")
    parser.add_argument("--limit-variant", choices=["rounds", "samples"], default="rounds",
                        help="rounds: a turn budget (--max-turns); samples: a total sample budget (--max-total-samples)")
    parser.add_argument("--max-turns", type=int, default=4,
                        help="Turn budget for --limit-variant rounds (default: 4, the paper's protocol)")
    parser.add_argument("--max-total-samples", type=int, default=None,
                        help="Total sample budget per question for --limit-variant samples")
    parser.add_argument("--max-samples-per-query", type=int, default=None,
                        help="Optional per-query sample budget for --limit-variant samples")
    parser.add_argument("--sample-accounting", choices=["rows", "cells"], default="rows",
                        help="Accounting unit for sample budgets")
    parser.add_argument("--max-new-tokens", type=int, default=65536,
                        help="Agent output cap per call; set it to the model's maximum output length")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--run-name", default=None,
                        help="Shorthand: sets --output-dir to ./out/query/<run-name> and "
                             "-o to ./out/query/<run-name>/<run-name>.json. Explicit --output-dir / -o override.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSON path (default: <output-dir>/agent_<timestamp>.json)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for query CSVs and per-experiment logs (default: ./out/query)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable auto-skip of questions already completed in --output-dir/agent_logs/")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    resource_limits = resolve_resource_limits(args, parser)

    # --run-name shorthand: derive --output-dir and -o unless explicitly set.
    if args.run_name:
        if args.output_dir is None:
            args.output_dir = f"./out/query/{args.run_name}"
        if args.output is None:
            args.output = os.path.join(args.output_dir, f"{args.run_name}.json")
    if args.output_dir is None:
        args.output_dir = "./out/query"

    if args.scientist_backend == "bedrock" or args.world_model_backend == "bedrock":
        from bedrock_llm import BedrockLLM
    if args.scientist_backend == "openai" or args.world_model_backend == "openai":
        from openai_llm import OpenAILLM
    if args.scientist_backend == "gemini":
        from gemini_llm import GeminiLLM

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not (args.worlds_dir or args.world_json):
        parser.error("give --worlds-dir or --world-json")

    # ------------------------------------------------------------------
    # Load worlds
    # ------------------------------------------------------------------
    worlds = load_worlds(worlds_dir=args.worlds_dir, world_json=args.world_json)
    if not worlds:
        logger.error("No worlds found.")
        sys.exit(1)

    total_q = sum(len(w["questions"]) for _, w in worlds)
    logger.info(f"Loaded {len(worlds)} world(s) with {total_q} total question(s)")

    # ------------------------------------------------------------------
    # Prepare output dirs (needed before resume scan)
    # ------------------------------------------------------------------
    query_data_dir = os.path.join(args.output_dir, "agent_query_data")
    log_dir = os.path.join(args.output_dir, "agent_logs")
    os.makedirs(query_data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Resume support: scan log_dir BEFORE loading any LLM so we can skip
    # heavy model loading if everything is already done.
    # ------------------------------------------------------------------
    done_experiments: Dict = {} if args.no_resume else load_done_experiments(log_dir)
    if done_experiments:
        incompatible = [
            path for path, data in (v for v in done_experiments.values())
            if not resource_limits_compatible(resource_limits, data.get("resource_limits"))
            or data.get("scientist_model") != args.scientist_model
        ]
        if incompatible:
            logger.error(
                "Resume refused: found %d completed log(s) with a different "
                "model or resource limits in %s. Use a distinct --run-name or --no-resume.",
                len(incompatible),
                log_dir,
            )
            logger.error("First incompatible log: %s", incompatible[0])
            sys.exit(2)
        logger.info(f"Resume: found {len(done_experiments)} completed experiment log(s) in {log_dir} — will skip those questions")

    n_pending = sum(
        1
        for fp, w in worlds
        for q in (w.get("questions") or [])
        if (os.path.basename(fp), q.get("id")) not in done_experiments
    )
    logger.info(f"Pending questions to run: {n_pending}/{total_q}")

    # ------------------------------------------------------------------
    # Build world model LLM (configurable backend, used for query parsing)
    # ------------------------------------------------------------------
    if n_pending == 0:
        logger.info("Nothing to do — skipping LLM load")
        world_llm = None
        scientist_llm = None
        scientist_model = args.scientist_model
    else:
        logger.info(f"Loading world model LLM: {args.world_model} (backend={args.world_model_backend})")
        # Parser wants near-deterministic output; keep temperature low regardless of backend.
        if args.world_model_backend == "openai":
            world_llm = OpenAILLM(
                model_name=args.world_model,
                base_url=args.world_model_base_url,
                api_key=args.world_model_api_key,
                max_new_tokens=512,
                temperature=0.1,
            )
        else:
            world_llm = BedrockLLM(
                model_id=args.world_model,
                max_new_tokens=512,
                temperature=0.1,
            )

        # ------------------------------------------------------------------
        # Build scientist LLM (configurable)
        # ------------------------------------------------------------------
        scientist_model = args.scientist_model
        logger.info(f"Loading scientist LLM: {scientist_model} (backend={args.scientist_backend})")
        if args.scientist_backend == "openai":
            scientist_llm = OpenAILLM(
                model_name=scientist_model,
                base_url=args.scientist_base_url,
                api_key=args.scientist_api_key,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        elif args.scientist_backend == "bedrock":
            scientist_llm = BedrockLLM(
                model_id=scientist_model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        else:
            scientist_llm = GeminiLLM(
                model_id=scientist_model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    all_results: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    n_resumed = 0
    done = 0

    for wi, (filepath, world_data) in enumerate(worlds):
        world_name = Path(filepath).stem
        topic = world_data.get("meta", {}).get("topic", "")
        n_nodes = world_data.get("meta", {}).get("n_nodes", 0)
        topology = world_data.get("meta", {}).get("topology", "")
        n_questions_in_world = len(world_data.get("questions", []) or [])
        logger.info(f"[{wi + 1}/{len(worlds)}] {world_name} ({topic}, n={n_nodes})")

        # Quick pre-check: which question ids in this world are NOT done yet?
        world_basename = os.path.basename(filepath)
        pending_ids = {
            q.get("id") for q in (world_data.get("questions") or [])
            if (world_basename, q.get("id")) not in done_experiments
        }

        if not pending_ids:
            # Re-create result rows from the prior logs so the aggregated
            # output JSON still contains every question.
            for q in world_data.get("questions") or []:
                qid = q.get("id")
                key = (world_basename, qid)
                log_path, prior = done_experiments[key]
                done += 1
                logger.info(f"  [{done}/{total_q}] q{qid}: SKIP (already done in {os.path.basename(log_path)})")
                question = Question(
                    question_type=q["question_type"],
                    question_text=q.get("question", ""),
                    ground_truth=q.get("answer", ""),
                    metadata={k: v for k, v in q.items()
                              if k not in ("question_type", "question", "answer")},
                )
                all_results.append(result_from_prior_log(
                    prior, log_path, filepath, world_name, topic, n_nodes, topology, question,
                ))
                n_resumed += 1
            logger.info(f"  All {n_questions_in_world} questions already done — skipping world setup")
            continue

        try:
            world_model, config = setup_world_model(filepath, world_llm, query_data_dir, WorldModel)
        except Exception as e:
            import traceback
            reason = f"{type(e).__name__}: {e}"
            logger.error(f"SKIPPING world {world_name}: {reason}")
            skipped.append({
                "world_file": filepath,
                "world_name": world_name,
                "topic": topic,
                "n_nodes": n_nodes,
                "topology": topology,
                "n_questions_skipped": len(pending_ids),
                "stage": "world_setup",
                "error": reason,
                "traceback": traceback.format_exc(),
            })
            done += n_questions_in_world
            continue

        questions = questions_from_config(config)

        for question in questions:
            done += 1
            q_id = question.metadata.get("id", 0)

            # Resume: if a log already exists for this (world, qid), skip and
            # re-emit the prior result row.
            key = (world_basename, q_id)
            if key in done_experiments:
                log_path, prior = done_experiments[key]
                logger.info(f"  [{done}/{total_q}] q{q_id}: SKIP (already done in {os.path.basename(log_path)})")
                all_results.append(result_from_prior_log(
                    prior, log_path, filepath, world_name, topic, n_nodes, topology, question,
                ))
                n_resumed += 1
                continue

            logger.info(f"  [{done}/{total_q}] q{q_id}: {question.question_text[:80]}...")

            try:
                # Fresh agent per question
                scientist = AGENTS[args.agent_type](llm=scientist_llm)

                orchestrator = Orchestrator(
                    world_model=world_model,
                    scientist=scientist,
                    question=question,
                    max_queries=args.max_queries,
                    log_dir=log_dir,
                    scientist_model=scientist_model,
                    world_model_name=args.world_model,
                    dataset_file=filepath,
                    limit_variant=resource_limits["limit_variant"],
                    max_turns=resource_limits["max_turns"],
                    max_total_samples=resource_limits["max_total_samples"],
                    max_samples_per_query=resource_limits["max_samples_per_query"],
                    sample_accounting=resource_limits["sample_accounting"],
                )

                exp_result = orchestrator.run()

                # Same result format as run_guess.py
                result = {
                    "world_file": filepath,
                    "world_name": world_name,
                    "topic": topic,
                    "n_nodes": n_nodes,
                    "topology": topology,
                    "question_id": q_id,
                    "question_text": question.question_text,
                    "question_type": question.question_type,
                    "difficulty": question.metadata.get("difficulty", ""),
                    "ground_truth": question.ground_truth,
                    "raw_response": (exp_result.turns[-1].raw_llm_response
                                     if exp_result.turns else ""),
                    "extracted_answer": exp_result.scientist_answer,
                    # Agent-specific extras (not used by the evaluators)
                    "total_queries": exp_result.total_queries,
                    "max_queries": exp_result.max_queries,
                    "resource_limits": exp_result.resource_limits,
                    "resource_usage": exp_result.resource_usage,
                }
                all_results.append(result)
                logger.info(f"  Answer: {exp_result.scientist_answer[:100]}")

            except Exception as e:
                import traceback
                reason = f"{type(e).__name__}: {e}"
                logger.error(f"SKIPPING {world_name} q{q_id}: {reason}")
                skipped.append({
                    "world_file": filepath,
                    "world_name": world_name,
                    "topic": topic,
                    "n_nodes": n_nodes,
                    "topology": topology,
                    "question_id": q_id,
                    "question_text": question.question_text,
                    "n_questions_skipped": 1,
                    "stage": "orchestrator_run",
                    "error": reason,
                    "traceback": traceback.format_exc(),
                })

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if args.output:
        out_path = args.output
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.output_dir, f"agent_{timestamp}.json")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "model": scientist_model,
        "world_model": args.world_model,
        "scientist_backend": args.scientist_backend,
        "method": args.agent_type,
        "max_queries": args.max_queries,
        "limit_variant": resource_limits["limit_variant"],
        "resource_limits": resource_limits,
        "timestamp": datetime.now().isoformat(),
        "n_worlds": len(worlds),
        "n_questions": len(all_results),
        "n_resumed": n_resumed,
        "n_skipped": sum(s["n_questions_skipped"] for s in skipped),
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Write skip report if anything was skipped
    if skipped:
        skip_path = str(Path(out_path).with_suffix("")) + "_skipped.json"
        with open(skip_path, "w") as f:
            json.dump(skipped, f, indent=2)
        n_skipped_q = sum(s["n_questions_skipped"] for s in skipped)
        logger.error(f"{len(skipped)} skip(s) ({n_skipped_q} questions). Details: {skip_path}")
        print(f"\nWARNING: {n_skipped_q} questions skipped. See {skip_path}")

    logger.info(f"Results saved to {out_path}")
    n_new = len(all_results) - n_resumed
    print(f"\nDone. {len(all_results)}/{total_q} questions answered "
          f"({n_new} new this run, {n_resumed} resumed from prior logs). Results: {out_path}")
    scorer = ("evaluate_decision.py" if any(r["question_type"].startswith("advanced_") for r in all_results)
              else "evaluate_struct.py")
    print(f"Score with: python {os.path.relpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), scorer))} "
          f"{out_path} --details")
    if skipped:
        sys.exit(1)


if __name__ == "__main__":
    main()
