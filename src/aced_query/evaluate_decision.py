"""evaluate_decision.py: score answers to the causal-decision questions.

One question per world, in five decision patterns (safety-constrained choice, mediator structure, satisficing,
subgroup-robust choice, invalid premise). Each question carries a `metadata.gold` block with everything needed to
score an answer (effects, deltas, candidate sets, baselines).

For each result row:
    1. Load the world JSON to get the variable list and the gold metadata.
    2. Ask the extraction LLM (Bedrock) for a structured, pattern-specific JSON reading of the answer.
    3. Score the extraction against the gold metadata with pattern-specific tolerance bands.

Usage:
    python evaluate_decision.py RUN.json [MORE.json ...] --details -o eval.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# The answer extractor used for every number in the paper.
EXTRACT_MODEL = "us.anthropic.claude-opus-4-6-v1"


# ---------------------------------------------------------------------------
# Default scoring tolerances (command-line options; used for every number in the paper)
# ---------------------------------------------------------------------------

DEFAULT_TIE_TOLERANCE = 0.05            # E[target] tie window for argmin
DEFAULT_SAFETY_TOLERANCE = 0.05         # delta_safety ≤ this counts as "safe" at eval time
DEFAULT_SUBGROUP_MIN_IMPROVEMENT = 0.15  # robust answer must beat this in EVERY subgroup


# ---------------------------------------------------------------------------
# Bedrock-backed extractor
# ---------------------------------------------------------------------------

@dataclass
class BedrockExtractor:
    model_id: str
    region_name: Optional[str] = None
    temperature: float = 0.0
    max_new_tokens: int = 600

    def __post_init__(self):
        from bedrock_llm import BedrockLLM
        self._llm = BedrockLLM(
            model_id=self.model_id,
            region_name=self.region_name,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )
        logger.info(f"Extractor ready: {self.model_id}")

    def extract_json(self, system: str, user: str, max_tries: int = 3) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for _ in range(max_tries):
            raw = self._llm.generate(system, user)
            try:
                return _first_balanced_json(raw)
            except Exception as e:
                last_err = e
        raise RuntimeError(f"extractor JSON parse failed: {last_err}")


def _first_balanced_json(text: str) -> Dict[str, Any]:
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return json.loads(text[start : i + 1])
    raise ValueError("No balanced JSON object found")


# ---------------------------------------------------------------------------
# Archetype-specific extraction prompts
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = (
    "You are a precise answer extractor. The scientist responded verbosely; "
    "you must extract their FINAL answer in the exact JSON shape requested. "
    "Output only the JSON object — no prose, no markdown, no commentary."
)


def _extract_safety_constrained(
    extractor: BedrockExtractor, question: str, response: str,
    intervenable: List[Dict[str, Any]],
) -> Dict[str, Any]:
    var_list = "\n".join(
        f"  - {v['name']} (values: {', '.join(v['values'])})"
        for v in intervenable
    )
    user = f"""Question:
{question}

Manipulable variables and their states:
{var_list}

Scientist's response:
{response}

Extract the SINGLE intervention they recommend (Variable = value, using the
exact variable name and value from the list above). Output exactly:

{{"variable": "<VarName>", "value": "<state>"}}

If they explicitly recommend none, output: {{"variable": null, "value": null}}.
"""
    return extractor.extract_json(_EXTRACT_SYSTEM, user)


def _extract_mediator_structure(
    extractor: BedrockExtractor, question: str, response: str,
    sub_variant: str, candidates: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if sub_variant == "which_mediator":
        cand_str = ", ".join(candidates or [])
        user = f"""Question:
{question}

Candidate variables: {cand_str}

Scientist's response:
{response}

Extract the SINGLE variable they identify as the true mediator. Output:
{{"chosen_mediator": "<VarName>"}}
Use the exact variable name. If they refuse to choose, use null.
"""
        return extractor.extract_json(_EXTRACT_SYSTEM, user)
    user = f"""Question:
{question}

Scientist's response:
{response}

Extract their classification of the mediator structure. Use one of EXACTLY
these three labels:
  - "only_through_M"        — T affects O only via M.
  - "also_direct_or_other"  — T affects O via M and also directly/elsewhere.
  - "not_mediator"          — M does not lie on a directed path from T to O.

Output exactly: {{"label": "<one of the three>"}}
"""
    return extractor.extract_json(_EXTRACT_SYSTEM, user)


def _extract_satisficing(
    extractor: BedrockExtractor, question: str, response: str,
    intervenable: List[Dict[str, Any]],
) -> Dict[str, Any]:
    var_list = "\n".join(
        f"  - {v['name']} (values: {', '.join(v['values'])})"
        for v in intervenable
    )
    user = f"""Question:
{question}

Manipulable variables and their states:
{var_list}

Scientist's response:
{response}

Extract their final answer:
  - If they identified one or more interventions that meaningfully improve
    the target, list them as objects {{"variable": ..., "value": ...}}.
  - If they explicitly answered 'none' (no intervention works), set the
    list to [].

Output exactly:
{{"interventions": [{{"variable": "<VarName>", "value": "<state>"}}, ...],
  "claim_none": <true|false>}}

claim_none should be true ONLY if they explicitly said no intervention works.
"""
    return extractor.extract_json(_EXTRACT_SYSTEM, user)


def _extract_subgroup_robust(
    extractor: BedrockExtractor, question: str, response: str,
    intervenable: List[Dict[str, Any]],
) -> Dict[str, Any]:
    var_list = "\n".join(
        f"  - {v['name']} (values: {', '.join(v['values'])})"
        for v in intervenable
    )
    user = f"""Question:
{question}

Manipulable variables and their states:
{var_list}

Scientist's response:
{response}

The scientist was asked which intervention reliably helps EVERY subgroup,
not just the population average. Extract their single recommendation.

Output exactly:
{{"variable": "<VarName>", "value": "<state>"}}
"""
    return extractor.extract_json(_EXTRACT_SYSTEM, user)


def _extract_invalid_premise(
    extractor: BedrockExtractor, question: str, response: str,
    intervenable: List[Dict[str, Any]],
) -> Dict[str, Any]:
    var_list = "\n".join(
        f"  - {v['name']} (values: {', '.join(v['values'])})"
        for v in intervenable
    )
    user = f"""Question:
{question}

Manipulable variables and their states:
{var_list}

Scientist's response:
{response}

The scientist was asked (a) is the proposed intervention valid/sensible? and
(b) if not, name a manipulable alternative.

Output exactly:
{{"is_valid_intervention": <true|false>,
  "alternative": {{"variable": "<VarName>", "value": "<state>"}}}}

If they say the proposed intervention is valid/sensible, set
"is_valid_intervention" to true. If they did not name an alternative, set
"alternative" to null.
"""
    return extractor.extract_json(_EXTRACT_SYSTEM, user)


# ---------------------------------------------------------------------------
# Per-archetype scoring against lazy-gold metadata
# ---------------------------------------------------------------------------

def _norm(s: Any) -> str:
    return str(s).strip().lower()


def _ci_match(a: str, b: str) -> bool:
    return _norm(a) == _norm(b)


def _score_safety_constrained(
    extracted: Dict[str, Any], gold: Dict[str, Any],
    tie_tolerance: float, safety_tolerance: float,
) -> Dict[str, Any]:
    var = extracted.get("variable")
    val = extracted.get("value")
    if not var or val is None:
        return {"correct": False, "detail": "no intervention extracted",
                "metrics": {}, "answer_parsed": extracted}

    safe_set = gold["all_safe_improvers"]
    rejected = gold["rejected_risky"]

    # Acceptable if the (var, val) is in the safe-improvers list within
    # tie_tolerance of the BEST safe action (counts the Pareto-acceptable
    # safe actions, not just the strict argmin).
    best = gold["best_safe_action"]
    best_delta = next((r["delta_target"] for r in safe_set
                       if r["variable"] == best["variable"]
                       and _ci_match(r["value"], best["value"])), None)

    matched = next(
        (r for r in safe_set
         if r["variable"] == var and _ci_match(r["value"], val)),
        None,
    )
    rejected_match = next(
        (r for r in rejected
         if r["variable"] == var and _ci_match(r["value"], val)),
        None,
    )
    if matched is not None:
        # Safety check: delta_safety must not exceed the eval-time tolerance.
        safety_ok = matched["delta_safety"] <= safety_tolerance
        # Within tolerance of best target effect?
        within = (best_delta is None
                  or matched["delta_target"] - best_delta <= tie_tolerance)
        correct = within and safety_ok
        best_delta_str = f"{best_delta:+.3f}" if best_delta is not None else "n/a"
        detail = (f"Pred {var}={val}: Δtarget={matched['delta_target']:+.3f}, "
                  f"Δsafety={matched['delta_safety']:+.3f}; "
                  f"best safe Δtarget={best_delta_str}")
        if not safety_ok:
            detail += f" — REJECTED: Δsafety={matched['delta_safety']:+.3f} > ε={safety_tolerance}"
        return {
            "correct": correct,
            "detail": detail,
            "metrics": {
                "exact_best": (var == best["variable"]
                               and _ci_match(val, best["value"])),
                "in_safe_pareto_set": True,
                "safety_ok": safety_ok,
                "delta_target": matched["delta_target"],
                "delta_safety": matched["delta_safety"],
                "best_delta_target": best_delta,
            },
            "answer_parsed": {"variable": var, "value": val},
        }
    if rejected_match is not None:
        return {
            "correct": False,
            "detail": (f"Pred {var}={val} improves target but worsens safety "
                       f"(Δsafety={rejected_match['delta_safety']:+.3f})"),
            "metrics": {"in_safe_pareto_set": False, "rejected_for_safety": True,
                        "delta_target": rejected_match["delta_target"],
                        "delta_safety": rejected_match["delta_safety"]},
            "answer_parsed": {"variable": var, "value": val},
        }
    return {
        "correct": False,
        "detail": f"Pred {var}={val} not among meaningful interventions",
        "metrics": {"in_safe_pareto_set": False},
        "answer_parsed": {"variable": var, "value": val},
    }


def _score_mediator_structure(
    extracted: Dict[str, Any], gold: Dict[str, Any], sub_variant: str,
) -> Dict[str, Any]:
    if sub_variant == "which_mediator":
        chosen = extracted.get("chosen_mediator")
        gold_var = gold["true_mediator"]
        correct = bool(chosen) and chosen == gold_var
        return {
            "correct": correct,
            "detail": f"Pred mediator={chosen}; gold={gold_var}",
            "metrics": {"sub_variant": sub_variant},
            "answer_parsed": {"chosen_mediator": chosen},
        }
    label = _norm(extracted.get("label"))
    gold_label = _norm(gold["label"])
    correct = (label == gold_label)
    return {
        "correct": correct,
        "detail": f"Pred label={label}; gold={gold_label}",
        "metrics": {"sub_variant": sub_variant},
        "answer_parsed": {"label": label},
    }


def _score_satisficing(
    extracted: Dict[str, Any], gold: Dict[str, Any],
) -> Dict[str, Any]:
    feasible = gold["feasible_actions"]
    feasible_set = {(r["variable"], _norm(r["value"])) for r in feasible}

    if extracted.get("claim_none"):
        # If actually empty, correct; otherwise wrong
        correct = (len(feasible_set) == 0)
        return {
            "correct": correct,
            "detail": (f"Predicted 'none'; "
                       f"{'no feasible interventions exist' if correct else f'{len(feasible_set)} feasible exist'}"),
            "metrics": {"feasible_set_size": len(feasible_set), "claim_none": True},
            "answer_parsed": {"claim_none": True},
        }

    pred = extracted.get("interventions") or []
    pred_pairs = [(r.get("variable"), _norm(r.get("value")))
                  for r in pred
                  if r.get("variable") and r.get("value") is not None]
    # ANY feasible action counts as correct (satisficing semantics)
    matches = [p for p in pred_pairs if p in feasible_set]
    correct = len(matches) > 0 if feasible_set else False
    return {
        "correct": correct,
        "detail": (f"Pred {pred_pairs}; feasible {sorted(feasible_set)} "
                   f"({'match' if matches else 'no match'})"),
        "metrics": {
            "feasible_set_size": len(feasible_set),
            "n_predicted": len(pred_pairs),
            "n_correct_in_pred": len(matches),
            "claim_none": False,
        },
        "answer_parsed": pred_pairs,
    }


def _score_subgroup_robust(
    extracted: Dict[str, Any], gold: Dict[str, Any], min_improvement: float,
) -> Dict[str, Any]:
    var = extracted.get("variable")
    val = extracted.get("value")
    rob = gold["robust_action"]
    avg = gold["avg_best_but_uneven"]
    bad = gold["harmful_in_subgroup"]

    # Acceptable: ANY (var, val) whose min_improvement across ALL subgroups
    # meets the eval-time threshold.  Gold-only-exact-match would mark agents
    # wrong for picking a different-but-equally-robust action.
    all_acceptable = gold.get("acceptable_robust_actions") or [rob]
    # Apply eval-time strictness: require min_improvement >= threshold.
    acceptable = [a for a in all_acceptable
                  if a.get("min_improvement", 0.0) >= min_improvement]
    if not acceptable:
        # Fall back to gold if nothing clears the bar (edge case)
        acceptable = [rob] if rob.get("min_improvement", 0.0) >= min_improvement else []
    acceptable_set = {(a["variable"], _norm(a["value"])) for a in acceptable}
    pair_norm = (var, _norm(val) if val is not None else "")
    acceptable_match = pair_norm in acceptable_set

    exact_gold = (var == rob["variable"] and _ci_match(val, rob["value"]))
    chose_avg = (var == avg["variable"] and _ci_match(val, avg["value"]))
    chose_bad = (var == bad["variable"] and _ci_match(val, bad["value"]))

    return {
        "correct": acceptable_match,
        "detail": (f"Pred {var}={val}; gold robust={rob['variable']}={rob['value']} "
                   f"(min_imp={rob['min_improvement']:+.3f}); "
                   f"trap-avg={avg['variable']}={avg['value']} "
                   f"(low_imp={avg['improvement_low']:+.3f}, "
                   f"high_imp={avg['improvement_high']:+.3f}); "
                   f"acceptable_set_size={len(acceptable_set)} "
                   f"(after min_imp≥{min_improvement} filter)"),
        "metrics": {
            "exact_match_gold": exact_gold,
            "in_acceptable_set": acceptable_match,
            "chose_avg_best_trap": chose_avg,
            "chose_harmful_subgroup_action": chose_bad,
            "robust_min_improvement": rob["min_improvement"],
            "n_acceptable_robust_actions": len(acceptable_set),
        },
        "answer_parsed": {"variable": var, "value": val},
    }


def _score_invalid_premise(
    extracted: Dict[str, Any], gold: Dict[str, Any], world: Dict[str, Any],
) -> Dict[str, Any]:
    valid = bool(extracted.get("is_valid_intervention"))
    gold_valid = bool(gold.get("is_valid_intervention", False))
    judgment_correct = (valid == gold_valid)

    alt = extracted.get("alternative") or {}
    alt_var = alt.get("variable") if isinstance(alt, dict) else None
    alt_val = alt.get("value") if isinstance(alt, dict) else None

    intervenable_names = {
        v["name"] for v in world["variables"] if v.get("intervenable")
    }
    alt_is_intervenable = alt_var in intervenable_names if alt_var else False

    # Tight check: the named (var, value) pair must MEANINGFULLY shift the
    # target in the right direction.  `acceptable_alternatives` was computed
    # at world-generation time as every (intervenable_var, value) whose
    # do-effect on the target exceeds the accept_threshold (≈ MIN_EFFECT/2).
    acceptable = gold.get("acceptable_alternatives") or []
    acceptable_set = {
        (a["variable"], _norm(a["value"])) for a in acceptable
    }
    pair_norm = (alt_var, _norm(alt_val) if alt_val is not None else "")
    alt_is_acceptable = (pair_norm in acceptable_set) if alt_var else False

    gold_alt = gold.get("alternative") or {}
    alt_is_exact_gold = (alt_var == gold_alt.get("variable")
                         and _ci_match(alt_val or "", gold_alt.get("value") or ""))

    if gold_valid:
        correct = judgment_correct
        proposed = gold.get("proposed_intervention") or {}
        return {
            "correct": correct,
            "detail": (f"Pred valid={valid} (gold=True); "
                       f"gold proposed={proposed.get('variable')}="
                       f"{proposed.get('value')}; "
                       f"extra_alt={alt_var}={alt_val}"),
            "metrics": {
                "judgment_correct": judgment_correct,
                "gold_is_valid_intervention": True,
                "provided_unneeded_alternative": bool(alt_var),
                "n_acceptable_alternatives": len(acceptable_set),
            },
            "answer_parsed": {
                "is_valid_intervention": valid,
                "alternative": {"variable": alt_var, "value": alt_val},
            },
        }

    correct = judgment_correct and alt_is_acceptable
    return {
        "correct": correct,
        "detail": (f"Pred valid={valid} (gold=False); "
                   f"alt={alt_var}={alt_val} "
                   f"(intervenable={alt_is_intervenable}, "
                   f"acceptable={alt_is_acceptable}, "
                   f"exact_gold={alt_is_exact_gold})"),
        "metrics": {
            "judgment_correct": judgment_correct,
            "gold_is_valid_intervention": False,
            "alternative_intervenable": alt_is_intervenable,
            "alternative_meaningfully_helps": alt_is_acceptable,
            "alternative_matches_gold_exactly": alt_is_exact_gold,
            "n_acceptable_alternatives": len(acceptable_set),
        },
        "answer_parsed": {"is_valid_intervention": valid,
                          "alternative": {"variable": alt_var, "value": alt_val}},
    }


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

_ARCHETYPE_FROM_QTYPE = {
    "advanced_safety_constrained":   "safety_constrained",
    "advanced_mediator_structure":   "mediator_structure",
    "advanced_satisficing":          "satisficing",
    "advanced_subgroup_robust":      "subgroup_robust",
    "advanced_invalid_premise":      "invalid_premise",
}


def evaluate_one(
    extractor: BedrockExtractor, result: Dict[str, Any], world: Dict[str, Any],
    tie_tolerance: float, safety_tolerance: float,
    subgroup_min_improvement: float,
) -> Dict[str, Any]:
    qtype = result.get("question_type", "")
    archetype = _ARCHETYPE_FROM_QTYPE.get(qtype)
    if not archetype:
        return {
            "correct": False,
            "detail": f"unrecognized question_type {qtype!r}",
            "metrics": {}, "answer_parsed": result.get("extracted_answer", "")[:80],
        }

    question_id = result["question_id"]
    question = next((q for q in world["questions"]
                     if q.get("id") == question_id), None)
    if question is None:
        return {"correct": False, "detail": f"question_id {question_id} not in world",
                "metrics": {}, "answer_parsed": ""}

    metadata = question.get("metadata") or {}
    gold = metadata.get("gold") or {}
    sub_variant = metadata.get("sub_variant")
    intervenable = [v for v in world["variables"] if v.get("intervenable")]

    # A give-up is not an answer: it scores 0, as in the structural scorer, whatever its reasoning mentions.
    answer = str(result.get("extracted_answer") or "")
    if answer.startswith("GIVE_UP"):
        return {"correct": False, "detail": "Empty/give-up response (the agent gave up)",
                "metrics": {}, "answer_parsed": answer[:80], "archetype": archetype, "sub_variant": sub_variant}

    raw_response = result.get("raw_response") or result.get("extracted_answer") or ""
    question_text = question.get("question") or result.get("question_text", "")

    # Per-archetype extraction
    try:
        if archetype == "safety_constrained":
            extracted = _extract_safety_constrained(
                extractor, question_text, raw_response, intervenable,
            )
            score = _score_safety_constrained(
                extracted, gold, tie_tolerance, safety_tolerance,
            )
        elif archetype == "mediator_structure":
            cands = gold.get("candidates")
            extracted = _extract_mediator_structure(
                extractor, question_text, raw_response, sub_variant or "", cands,
            )
            score = _score_mediator_structure(extracted, gold, sub_variant or "")
        elif archetype == "satisficing":
            extracted = _extract_satisficing(
                extractor, question_text, raw_response, intervenable,
            )
            score = _score_satisficing(extracted, gold)
        elif archetype == "subgroup_robust":
            extracted = _extract_subgroup_robust(
                extractor, question_text, raw_response, intervenable,
            )
            score = _score_subgroup_robust(extracted, gold, subgroup_min_improvement)
        elif archetype == "invalid_premise":
            extracted = _extract_invalid_premise(
                extractor, question_text, raw_response, intervenable,
            )
            score = _score_invalid_premise(extracted, gold, world)
        else:
            return {"correct": False, "detail": f"no scorer for {archetype}",
                    "metrics": {}, "answer_parsed": ""}
    except Exception as e:
        return {
            "correct": False,
            "detail": f"extraction/scoring error: {type(e).__name__}: {e}",
            "metrics": {"error": str(e)},
            "answer_parsed": "",
        }

    score["archetype"] = archetype
    score["sub_variant"] = sub_variant
    return score


# ---------------------------------------------------------------------------
# Aggregate scoring
# ---------------------------------------------------------------------------

def _summarize(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(items)
    c = sum(1 for it in items if it["eval"]["correct"])
    return {
        "total": n,
        "correct": c,
        "accuracy": round(c / n, 3) if n else 0.0,
    }


def compute_scores(evaluated: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores: Dict[str, Any] = {}
    scores["overall"] = _summarize(evaluated)

    by_arch: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_arch_sub: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for e in evaluated:
        arch = e["eval"].get("archetype") or "unknown"
        sub = e["eval"].get("sub_variant")
        by_arch[arch].append(e)
        if sub:
            by_arch_sub[(arch, sub)].append(e)

    scores["by_archetype"] = {a: _summarize(items) for a, items in sorted(by_arch.items())}
    if by_arch_sub:
        scores["by_archetype_sub"] = {
            f"{a}/{s}": _summarize(items)
            for (a, s), items in sorted(by_arch_sub.items())
        }

    by_topic: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in evaluated:
        by_topic[e.get("topic", "?")].append(e)
    scores["by_topic"] = {t: _summarize(items) for t, items in sorted(by_topic.items())}

    by_n: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for e in evaluated:
        by_n[int(e.get("n_nodes", 0))].append(e)
    scores["by_n_nodes"] = {str(n): _summarize(items) for n, items in sorted(by_n.items())}

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_results(paths: List[str]) -> Tuple[List[Dict[str, Any]], str]:
    rows: List[Dict[str, Any]] = []
    model_str = "?"
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        model_str = d.get("model") or d.get("run_metadata", {}).get("model", model_str)
        rows.extend(d.get("results", []))
    return rows, model_str


def main():
    parser = argparse.ArgumentParser(
        description="Score answers to the causal-decision questions (LLM extraction always on).",
    )
    parser.add_argument("results_json", nargs="+",
                        help="Path(s) to results JSON(s).")
    parser.add_argument("-o", "--output", default=None,
                        help="Save evaluation to this path.")
    parser.add_argument("--extract-model", default=EXTRACT_MODEL,
                        help=f"Bedrock model id of the answer extractor (default: {EXTRACT_MODEL})")
    parser.add_argument("--region", default=None,
                        help="AWS region for Bedrock; defaults to env or us-west-2.")
    parser.add_argument("--tie-tolerance", type=float, default=DEFAULT_TIE_TOLERANCE)
    parser.add_argument("--safety-tolerance", type=float, default=DEFAULT_SAFETY_TOLERANCE)
    parser.add_argument("--subgroup-min-improvement", type=float,
                        default=DEFAULT_SUBGROUP_MIN_IMPROVEMENT)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    rows, model_str = _load_results(args.results_json)
    print(f"Loaded {len(rows)} result rows. Run model: {model_str}")
    print(f"Extraction model: {args.extract_model}")

    extractor = BedrockExtractor(model_id=args.extract_model, region_name=args.region)

    world_cache: Dict[str, Dict[str, Any]] = {}
    evaluated: List[Dict[str, Any]] = []
    for ri, r in enumerate(rows):
        wf = r["world_file"]
        if wf not in world_cache:
            with open(wf) as f:
                world_cache[wf] = json.load(f)
        world = world_cache[wf]
        topic = world.get("meta", {}).get("topic", "")
        n_nodes = int(world.get("meta", {}).get("n_nodes", 0))
        ev = evaluate_one(
            extractor, r, world,
            tie_tolerance=args.tie_tolerance,
            safety_tolerance=args.safety_tolerance,
            subgroup_min_improvement=args.subgroup_min_improvement,
        )
        evaluated.append({
            "world_name": r.get("world_name"),
            "world_file": wf,
            "topic": topic,
            "n_nodes": n_nodes,
            "question_id": r.get("question_id"),
            "question_type": r.get("question_type"),
            "question_text": r.get("question_text"),
            "raw_response": r.get("raw_response", ""),
            "extracted_answer": r.get("extracted_answer", ""),
            "eval": ev,
        })
        if (ri + 1) % 10 == 0 or (ri + 1) == len(rows):
            n_correct = sum(1 for e in evaluated if e["eval"]["correct"])
            print(f"  [{ri + 1}/{len(rows)}] running accuracy: {n_correct}/{ri + 1}")

    scores = compute_scores(evaluated)

    print("\n" + "=" * 60)
    print("OVERALL")
    print("=" * 60)
    ov = scores["overall"]
    print(f"  Accuracy: {ov['correct']}/{ov['total']} = {ov['accuracy']:.1%}")

    print(f"\nBY ARCHETYPE:")
    for arch, s in scores["by_archetype"].items():
        print(f"  {arch:30s} {s['correct']}/{s['total']} = {s['accuracy']:.1%}")

    if "by_archetype_sub" in scores:
        print(f"\nBY ARCHETYPE/SUB-VARIANT:")
        for k, s in scores["by_archetype_sub"].items():
            print(f"  {k:38s} {s['correct']}/{s['total']} = {s['accuracy']:.1%}")

    if scores["by_topic"]:
        print(f"\nBY TOPIC:")
        for topic, s in scores["by_topic"].items():
            print(f"  {topic:30s} {s['correct']}/{s['total']} = {s['accuracy']:.1%}")

    print(f"\nBY N_NODES:")
    for n, s in scores["by_n_nodes"].items():
        print(f"  n={n:3s}  {s['correct']}/{s['total']} = {s['accuracy']:.1%}")

    if args.details:
        print(f"\n{'=' * 60}\nPER-QUESTION DETAILS\n{'=' * 60}")
        for e in evaluated:
            status = "CORRECT" if e["eval"]["correct"] else "WRONG"
            arch = e["eval"].get("archetype")
            sub = e["eval"].get("sub_variant")
            tag = (arch or "unknown") + (f"/{sub}" if sub else "")
            print(f"\n[{status}] {e['world_name']} ({tag})")
            print(f"  Q: {e['question_text'][:120]}")
            print(f"  {e['eval']['detail']}")

    if args.output:
        out_path = args.output
        source = args.results_json[0] if len(args.results_json) == 1 else args.results_json
        with open(out_path, "w") as f:
            json.dump({
                "source": source, "model": model_str,
                "extract_model": args.extract_model,
                "tie_tolerance": args.tie_tolerance,
                "safety_tolerance": args.safety_tolerance,
                "subgroup_min_improvement": args.subgroup_min_improvement,
                "scores": scores,
                "evaluated": evaluated,
            }, f, indent=2)
        print(f"\nEvaluation saved to {out_path}")


if __name__ == "__main__":
    main()
