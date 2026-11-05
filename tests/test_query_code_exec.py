"""The code agents' runner returns large outputs and large carried variables, and still stops runaway code."""
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src", "aced_query"))

import agent_active_code  # noqa: E402
import agent_modular  # noqa: E402

RUNNERS = [agent_active_code._execute_code, agent_modular._execute_code]


@pytest.mark.parametrize("run", RUNNERS)
def test_large_output_is_returned(run):
    out = run("print('a' * 200000)", {}, timeout=20)
    assert out.startswith("aaa") and "truncated" in out


@pytest.mark.parametrize("run", RUNNERS)
def test_large_variable_is_carried(run):
    ns = {}
    out = run("x = [str(i) for i in range(60000)]\nprint('ok')", ns, timeout=20)
    assert out.strip() == "ok" and len(ns["x"]) == 60000


@pytest.mark.parametrize("run", RUNNERS)
def test_runaway_code_times_out(run):
    out = run("while True:\n    pass", {}, timeout=3)
    assert out.startswith("[TIMEOUT")
