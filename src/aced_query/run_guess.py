"""
run_guess.py

Guess-only baseline: the agent's question prompt with no data access. Only the "answer" action is available (no
data queries, no code), so the model answers from the scenario and its domain knowledge alone.

Results are written to one file per difficulty group: <output>_<difficulty>.json.

Usage:
    python run_guess.py --worlds-dir ../../data/query/guess_struct --backend bedrock \
        --model us.anthropic.claude-opus-4-6-v1 -o ./out/query/guess_struct.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
import glob
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAI-compatible backend for the guess-only baseline
# ---------------------------------------------------------------------------

class OpenAIGuessLLM:
    """Guess-only LLM backed by an OpenAI-compatible API.

    Auto-applies per-model sampling presets from openai_llm
    (Qwen3.6 thinking mode, gpt-oss, DeepSeek V4 Pro) unless explicit kwargs override.
    Captures reasoning_content (chain of thought) into self.last_reasoning
    when the server exposes it (vLLM with --reasoning-parser).
    """
    def __init__(self, model_name: str,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 max_new_tokens: int = 1024,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 extra_body: Optional[Dict[str, Any]] = None,
                 use_preset: bool = True):
        from openai import OpenAI
        from openai_llm import resolve_preset

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        preset = resolve_preset(model_name) if use_preset else None
        self.temperature = temperature if temperature is not None else (
            preset["temperature"] if preset else 0.1)
        self.top_p = top_p if top_p is not None else (
            preset["top_p"] if preset else 1.0)
        self.extra_body = extra_body if extra_body is not None else (
            dict(preset["extra_body"]) if preset and "extra_body" in preset else None)

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        resolved_base = base_url or os.environ.get("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=resolved_key, base_url=resolved_base)
        self.last_reasoning: Optional[str] = None
        logger.info(
            f"OpenAI guess-only LLM ready — model={model_name}, "
            f"temperature={self.temperature}, top_p={self.top_p}, "
            f"extra_body={self.extra_body}, preset={'yes' if preset else 'no'}"
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        self.last_reasoning = getattr(msg, "reasoning_content", None)
        content = msg.content or ""
        return content.strip()


# ---------------------------------------------------------------------------
# Prompt construction — reduced scientist prompt (answer-only)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a discovery scientist. Your goal is to determine relationships between variables in an unknown causal graph.

You have NO access to data — you cannot request observational or interventional samples. You must reason from domain knowledge alone.

AVAILABLE ACTIONS:
1. ANSWER - Submit your final answer:
   - For Yes/No questions: answer "Yes" or "No"
   - For listing questions: provide the specific variable names in a list

CAUSAL DISCOVERY STRATEGY:

Think like a scientist reasoning about experiments.

You have two conceptual tools (but NO data access):
1. Observational data → shows correlations
2. Interventions do(X=value) → shows causal effects

Key ideas:

- Correlation alone cannot determine causation.
- To test if X causes Y:
    → change X (using do(X=...))
    → see if Y changes

Interpretation:
- If Y changes when X is changed → X causally affects Y
- If Y does NOT change → X does NOT cause Y
- If X and Y are correlated but no effect under intervention → likely common cause

Since you have no data, you must use your domain knowledge to reason about:
- Which variables are likely causally connected based on the domain
- What the likely causal directions are
- Whether variables are independent or dependent

Before answering, ensure:
- Your conclusion is based on careful causal reasoning about the domain
- You have considered alternative causal structures

OUTPUT FORMAT (required, in this exact order):
<reasoning>[Analysis, updated understanding, hypothesis, decision rationale]</reasoning>
<action type="answer">[Your final answer]</action>
Do NOT copy the example text — write your actual analysis and actual answer."""


def build_user_prompt(world: Dict[str, Any], question: Dict[str, Any]) -> str:
    """Build the user prompt from world JSON and a question dict.

    Mirrors the structure of ActiveAgent._get_decision_user_prompt() in agent_active.py, but with no query
    history, budget, or data sections (domain-knowledge reasoning only).
    """
    lines = []

    # SECTION 1: GOAL (same as scientist agent)
    lines.append("════════════════════════════════════════════════════════════════════════════════")
    lines.append(f"YOUR QUESTION: {question['question']}")
    lines.append("════════════════════════════════════════════════════════════════════════════════")
    lines.append("")

    # SECTION 2: VARIABLES (same as scientist agent — full catalog)
    lines.append("AVAILABLE VARIABLES (use exact names and state values for interventions):")
    for var in world["variables"]:
        states = ", ".join(var["values"])
        desc = var.get("desc", "")
        if desc:
            lines.append(f"  {var['name']}: {desc} (states: {states})")
        else:
            lines.append(f"  {var['name']} (states: {states})")
    lines.append("")

    story = world.get("story", "")
    if story:
        lines.append(f"CONTEXT: {story}")
        lines.append("")

    # SECTION 3: CONSTRAINTS (non-intervenable variables, matching scientist agent)
    if world.get("non_intervenable_variables"):
        ni_vars = world["non_intervenable_variables"]
        lines.append("INTERVENTION LIMITS — Cannot intervene on (non-manipulable variables):")
        if isinstance(ni_vars, list):
            for entry in ni_vars:
                lines.append(f"  - {entry['name']}: {entry['reason']}")
        elif isinstance(ni_vars, dict):
            for name, reason in ni_vars.items():
                lines.append(f"  - {name}: {reason}")
    else:
        lines.append("INTERVENTION LIMITS: All variables are intervenable")
    lines.append("")

    # SECTION 4: NO DATA NOTICE (replaces query history / latest result)
    lines.append("─── NO DATA AVAILABLE ───")
    lines.append("You have no access to observational or interventional data.")
    lines.append("You must reason from domain knowledge alone.")
    lines.append("")

    # ASSEMBLE (same closing as scientist agent)
    lines.append("=" * 80)
    lines.append("Now: Use your domain knowledge and causal reasoning to answer the question.")
    lines.append("Output: <reasoning> and <action type=\"answer\"> blocks (in that order).")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(response: str) -> str:
    """Extract the content from <action type="answer">...</action> tags."""
    # Strip think blocks (Qwen / DeepSeek-R1 style). Belt-and-suspenders even
    # when the server uses --reasoning-parser to route them out of .content.
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    match = re.search(
        r'<action\s+type="answer">\s*(.*?)\s*</action>',
        response, re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # Fallback: try <answer>...</answer>
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Last resort: last non-empty line
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    return lines[-1] if lines else response.strip()


def extract_reasoning(response: str) -> str:
    """Extract the content from <reasoning>...</reasoning> tags."""
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: everything before the <action> tag
    match = re.search(r'<action', response, re.IGNORECASE)
    if match:
        return response[:match.start()].strip()
    return response.strip()


# ---------------------------------------------------------------------------
# World loading
# ---------------------------------------------------------------------------

def load_worlds(worlds_dir: Optional[str] = None,
                world_json: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load world JSON files. Returns list of (filepath, parsed_json) tuples."""
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
# Main runner
# ---------------------------------------------------------------------------

def run_guess(worlds: List, llm: Any) -> List[Dict[str, Any]]:
    """Run the guess-only baseline on all worlds/questions. Returns list of result dicts."""
    all_results = []
    total_questions = sum(len(w["questions"]) for _, w in worlds)
    done = 0

    for filepath, world in worlds:
        world_name = Path(filepath).stem
        for question in world["questions"]:
            done += 1
            q_id = question.get("id", 0)
            logger.info(f"[{done}/{total_questions}] {world_name} q{q_id}: {question['question'][:80]}...")

            user_prompt = build_user_prompt(world, question)
            raw_response = llm.generate(SYSTEM_PROMPT, user_prompt)
            extracted = extract_answer(raw_response)

            result = {
                "world_file": filepath,
                "world_name": world_name,
                "topic": world.get("meta", {}).get("topic", ""),
                "n_nodes": world.get("meta", {}).get("n_nodes", 0),
                "topology": world.get("meta", {}).get("topology", ""),
                "question_id": q_id,
                "question_text": question["question"],
                "question_type": question.get("question_type", ""),
                "difficulty": question.get("difficulty", ""),
                "ground_truth": question["answer"],
                "raw_response": raw_response,
                "reasoning": extract_reasoning(raw_response),
                "extracted_answer": extracted,
            }
            all_results.append(result)
            logger.info(f"  Answer: {extracted[:100]}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Guess-only baseline (no data) for ACED-Query questions",
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("--worlds-dir", default=None,
                        help="Directory containing world_*.json files")
    parser.add_argument("--world-json", default=None,
                        help="Path to a single world JSON file")
    parser.add_argument("--backend", choices=["bedrock", "openai", "gemini"], default="bedrock",
                        help="'bedrock' = AWS Bedrock, 'openai' = OpenAI-compatible API (OpenAI, vLLM, ...), "
                             "'gemini' = Google AI Studio / Vertex")
    parser.add_argument("--model", required=True, help="Model id for that backend")
    parser.add_argument("--base-url", default=None,
                        help="API base URL for openai backend (falls back to OPENAI_BASE_URL)")
    parser.add_argument("--api-key", default=None,
                        help="API key for openai backend (falls back to OPENAI_API_KEY)")
    parser.add_argument("--max-new-tokens", type=int, default=65536,
                        help="Output cap per answer; set it to the model's maximum output length")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSON path prefix (default: ./out/query/guess_<timestamp>.json)")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not (args.worlds_dir or args.world_json):
        parser.error("give --worlds-dir or --world-json")

    # Load worlds
    worlds = load_worlds(worlds_dir=args.worlds_dir, world_json=args.world_json)
    if not worlds:
        logger.error("No worlds found.")
        sys.exit(1)

    total_q = sum(len(w["questions"]) for _, w in worlds)
    logger.info(f"Loaded {len(worlds)} world(s) with {total_q} total question(s)")

    # Load LLM
    if args.backend == "openai":
        llm = OpenAIGuessLLM(
            model_name=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    elif args.backend == "bedrock":
        from bedrock_llm import BedrockLLM
        llm = BedrockLLM(
            model_id=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    else:
        from gemini_llm import GeminiLLM
        llm = GeminiLLM(
            model_id=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    # Run
    results = run_guess(worlds, llm)

    # Save — split into one file per difficulty group
    if args.output:
        base = args.output[:-5] if args.output.endswith(".json") else args.output
    else:
        os.makedirs("./out/query", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"./out/query/guess_{timestamp}"

    Path(base).parent.mkdir(parents=True, exist_ok=True)

    groups: Dict[str, List] = defaultdict(list)
    for r in results:
        groups[r["difficulty"] or "unknown"].append(r)

    saved_paths = []
    run_ts = datetime.now().isoformat()
    for group_name in sorted(groups):
        group_results = groups[group_name]
        path = f"{base}_{group_name}.json"
        payload = {
            "run_metadata": {
                "model": args.model,
                "backend": args.backend,
                "worlds_source": args.worlds_dir or args.world_json,
                "timestamp": run_ts,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "question_group": group_name,
                "n_worlds": len(worlds),
                "n_questions": len(group_results),
                "prompt_type": "scientist_sub_prompt",
            },
            "results": group_results,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"  [{group_name}] {len(group_results)} questions → {path}")
        saved_paths.append(path)

    print(f"\nDone. {len(results)} questions answered across {len(groups)} difficulty group(s):")
    for p in saved_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
