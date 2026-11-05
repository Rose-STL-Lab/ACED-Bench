# ACED-Repair

## Running a model

```bash
bash run_experiments.sh repair NAME BACKEND MODEL [SPLIT]     # SPLIT: eval (54 worlds, default) or test (540 worlds)

# examples
MAX_TOKENS=128000 bash run_experiments.sh repair opus55 bedrock us.anthropic.claude-opus-5-5
OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \
    bash run_experiments.sh repair mymodel openai <served model name> test
```

Runs can be stopped and restarted; finished worlds are skipped (`--force` reruns them).
`python src/aced_repair/run_benchmark.py --help` lists every option.

## Protocol

15 experiments and 32 turns per world; the agent prompt is in `src/aced_repair/prompt.py`. Set `MAX_TOKENS` to the
model's maximum output length (the paper used 128000 for Claude Opus 5.5, Claude Sonnet 5.5, GPT-6 Astra and GPT-6.1
Sol, 8192 for Llama 4 Maverick, and 32768 for the other models).

Ablations in the paper: `--history full` (whole conversation every turn) and `--budget 30` / `--budget 8`.

## Output

`out/repair/NAME/` holds one `result_<world>.json` per world (the full trajectory and the grade) and `summary.json`
(fix rate, acceptance rate, mean mechanism score). In each grade:

| field | meaning |
|---|---|
| `benefit_recovered` | the fix's improvement of the true objective as a share of the oracle fix's improvement |
| `part_a_utility_ok` | the fix works (`benefit_recovered >= 0.9`) |
| `battery_fraction` | mechanism score: marker, confounded decoys, and signs of the true levers |
| `accepted` | the fix works and the mechanism score is at least 0.8 |
