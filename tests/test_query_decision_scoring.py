"""A give-up scores 0 in the decision scorer, without asking the extractor."""
import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src", "aced_query"))

import evaluate_decision as ed  # noqa: E402


class NoExtractor:
    def extract_json(self, *args, **kwargs):
        raise AssertionError("the extractor must not be called for a give-up")


def test_give_up_scores_zero():
    path = sorted(glob.glob(os.path.join(ROOT, "data", "query", "core_decision", "world_safety_constrained_*.json")))[0]
    world = json.load(open(path))
    q = world["questions"][0]
    result = {"question_type": q["question_type"], "question_id": q["id"],
              "extracted_answer": "GIVE_UP: Insufficient evidence",
              "raw_response": "The evidence favours the gold intervention, but I give up."}
    score = ed.evaluate_one(NoExtractor(), result, world, ed.DEFAULT_TIE_TOLERANCE,
                            ed.DEFAULT_SAFETY_TOLERANCE, ed.DEFAULT_SUBGROUP_MIN_IMPROVEMENT)
    assert score["correct"] is False and score["detail"].startswith("Empty/give-up")
    assert score["archetype"] == "safety_constrained"
