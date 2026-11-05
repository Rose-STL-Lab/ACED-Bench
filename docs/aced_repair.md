# ACED-Repair

The agent diagnoses a failing production system described in a situation report (about 25 controls and 26 measurable
signals). With 15 experiments it decides which controls to test, reads confounded and filtered data, and recommends a
fix with an account of the mechanism. The fix is re-simulated on the hidden true objective.

## Running a model

```bash
bash run_experiments.sh repair NAME BACKEND MODEL [SPLIT]     # SPLIT: eval (54 worlds, default) or test (540 worlds)

# examples
MAX_TOKENS=128000 bash run_experiments.sh repair opus55 bedrock us.anthropic.claude-opus-5-5
OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \
    bash run_experiments.sh repair mymodel openai <served model name> test
```

The script calls `src/aced_repair/run_benchmark.py`; `python src/aced_repair/run_benchmark.py --help` lists every
option and its default. Rerunning the same command skips worlds that already have a result (`--force` reruns them). A
world that was cut off restarts from its first turn; a world that fails (e.g. an API error after retries) is reported
and left for the next run.

`bash run_experiments.sh demo` runs a scripted agent that measures, analyses its data, tests the oracle fix and submits
the oracle answer on the 54 evaluation worlds; every world is accepted.

## Protocol

15 experiments and 32 turns per world. Each turn shows the situation report, the result of the last action and the
agent's own notes; `src/aced_repair/prompt.py` holds the agent prompt. Set `MAX_TOKENS` to the model's maximum output
length: the paper used 128000 for Claude Opus 5.5, Claude Sonnet 5.5, GPT-6 Astra and GPT-6.1 Sol, 8192 for Llama 4
Maverick, and 32768 for the other models.

Ablations in the paper: `--history full` sends the whole conversation every turn; `--budget 30` and `--budget 8`
change the experiment budget. `--no-resolver-llm` parses requests with the deterministic alias matcher alone (no
Bedrock calls for parsing).

## Output

- `out/repair/NAME/result_<world>.json`: the full trajectory (every turn, what the agent saw, how each request was
  parsed, code output) and the grade.
- `out/repair/NAME/data_<hash>/`: each world's experiment CSVs (hashed names, because world ids name the causal
  structure).
- `out/repair/NAME/summary.json`: fix rate, acceptance rate and mean mechanism score, with one row per world.

| grade field | meaning |
|---|---|
| `benefit_recovered` | improvement of the true objective under the recommended fix, as a share of the oracle fix's improvement (negative for a harmful fix; the paper clips it at 0) |
| `part_a_utility_ok` | the fix works: `benefit_recovered >= 0.9` (the fix rate counts these worlds) |
| `battery_fraction` | mechanism score: the mechanism proxy, the confounded decoys, and the signs of the true levers |
| `accepted` | the fix works and the mechanism score is at least 0.8 |

`recommended_intervention` and `gold_intervention` give each control's setting on an internal 0–100 scale of its range
(the agent answers in the physical units of the situation report).
