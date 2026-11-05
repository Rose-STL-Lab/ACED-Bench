"""ACED-Query without an API: one episode with a scripted LLM as both the request parser and the agent."""
import glob
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src", "aced_query"))

from agent_active import ActiveAgent  # noqa: E402
from evaluate_struct import evaluate_answer  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402
from run_agent_batch import questions_from_config, setup_world_model  # noqa: E402
from world_model import WorldModel  # noqa: E402


class ScriptedLLM:
    """Parses every request as 300 observational samples of all variables; the agent queries once, then answers."""

    def __init__(self, answer):
        self.answer = answer
        self.agent_calls = 0

    def generate(self, system_prompt, user_prompt, max_new_tokens=None):
        if "query parser" in system_prompt:
            return '<json>{"query_type": "observational", "n_samples": 300, "variables": null, "interventions": []}</json>'
        self.agent_calls += 1
        if self.agent_calls == 1:
            return '<action type="query">Give me 300 observational samples of all variables</action>'
        return f'<action type="answer">{self.answer}</action>'


def test_active_agent_episode(tmp_path):
    path = sorted(glob.glob(os.path.join(ROOT, "data", "query", "core_struct", "world_*.json")))[0]
    llm = ScriptedLLM(None)
    world_model, config = setup_world_model(path, llm, str(tmp_path / "data"), WorldModel)
    question = next(q for q in questions_from_config(config) if q.question_type == "causal_effect")
    answer = "Yes" if question.ground_truth in (True, "Yes", "yes") else "No"
    llm.answer = answer
    result = Orchestrator(world_model=world_model, scientist=ActiveAgent(llm=llm), question=question,
                          log_dir=str(tmp_path / "logs"), limit_variant="rounds", max_turns=4).run()
    assert result.total_queries == 1
    assert result.scientist_answer == answer
    assert evaluate_answer(result.scientist_answer, question.ground_truth)["correct"]
    assert len(glob.glob(str(tmp_path / "logs" / "experiment_*.json"))) == 1
