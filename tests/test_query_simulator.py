"""ACED-Query without an API: every evaluation world converts to a Bayesian network that can be sampled and intervened on."""
import glob
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src", "aced_query"))

from json_converter import convert_json_to_bif  # noqa: E402
from bn_simulator import BNSimulator  # noqa: E402


def test_worlds_sample(tmp_path):
    files = sorted(glob.glob(os.path.join(ROOT, "data", "query", "core_struct", "world_*.json")))
    files += sorted(glob.glob(os.path.join(ROOT, "data", "query", "core_decision", "world_*.json")))[:5]
    for f in files:
        bif = tmp_path / (os.path.basename(f)[:-5] + ".bif")
        convert_json_to_bif(f, str(bif))
        sim = BNSimulator.from_bif(str(bif))
        df = sim.sample_observational(200, seed=0)
        assert df.shape == (200, len(sim.get_nodes()))
        node = sim.get_nodes()[0]
        state = sim.get_state_names(node)[0]
        di = sim.sample_interventional({node: state}, 50, seed=0)
        assert (di[node] == state).all()
