"""The oracle fix and the scoring key stored with a world are what the oracle computes from the world's model."""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src", "aced_repair"))

from agent_loop import load_world_file  # noqa: E402
from oracle import counterfactual_battery, optimal_gold  # noqa: E402


def test_oracle_recomputes_stored_answers():
    path = os.path.join(ROOT, "data", "repair", "eval", "world_catalysis_confounded_chain_53000220.json.gz")
    world, stored = load_world_file(path)
    gold = optimal_gold(world)
    assert gold["intervention"] == stored["gold"]["intervention"]
    assert abs(gold["expected_utility"] - stored["gold"]["expected_utility"]) < 1e-9
    assert counterfactual_battery(world) == stored["battery"]
