"""Benchmark data: split sizes, the evaluation set is a subset of the test set, and every world file loads."""
import collections
import glob
import gzip
import json
import os

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def _index(split):
    return json.load(open(os.path.join(DATA, "repair", split, "index.json")))


def test_repair_split_sizes():
    for split, n in {"test": 540, "eval": 54}.items():
        idx = _index(split)
        assert len(idx) == n
        files = glob.glob(os.path.join(DATA, "repair", split, "world_*.json.gz"))
        assert sorted(os.path.basename(f) for f in files) == sorted(r["file"] for r in idx)


def test_repair_structures_balanced():
    for split, per in {"test": 60, "eval": 6}.items():
        counts = collections.Counter(r["archetype"] for r in _index(split))
        assert len(counts) == 9 and set(counts.values()) == {per}


def test_eval_is_subset_of_test():
    for f in glob.glob(os.path.join(DATA, "repair", "eval", "world_*.json.gz")):
        other = os.path.join(DATA, "repair", "test", os.path.basename(f))
        assert open(f, "rb").read() == open(other, "rb").read()


def test_repair_worlds_load():
    for f in glob.glob(os.path.join(DATA, "repair", "eval", "world_*.json.gz")):
        w = json.load(gzip.open(f))
        assert {"scenario", "scm", "ground_truth", "oracle"} <= set(w)
        assert {"gold", "counterfactual_battery"} <= set(w["oracle"])


def test_query_worlds_load():
    sizes = {"core_struct": 10, "core_decision": 60, "guess_struct": 30, "guess_decision": 48}
    for split, n in sizes.items():
        files = glob.glob(os.path.join(DATA, "query", split, "world_*.json"))
        assert len(files) == n
        for f in files:
            w = json.load(open(f))
            assert w["questions"] and w["cpds"] and w["variables"]
