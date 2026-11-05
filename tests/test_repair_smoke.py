"""ACED-Repair end to end without an API: a scripted agent that recommends the oracle fix must be accepted."""
import glob
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUN = os.path.join(ROOT, "src", "aced_repair", "run_benchmark.py")


def test_mock_agent_is_accepted(tmp_path):
    worlds = tmp_path / "worlds"
    worlds.mkdir()
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "repair", "eval", "world_*.json.gz")))[:3]:
        shutil.copy(f, worlds)
    out = tmp_path / "out"
    subprocess.run([sys.executable, RUN, "--worlds-dir", str(worlds), "--backend", "mock", "--outdir", str(out)],
                   check=True, capture_output=True)
    rows = [json.load(open(f)) for f in glob.glob(str(out / "result_*.json"))]
    assert len(rows) == 3
    summary = json.load(open(out / "summary.json"))
    assert summary["acceptance_rate"] == 1.0 and summary["fix_rate"] == 1.0
