# ACED: Active Causal Evidence and Decision Benchmark

Code and data for the NeurIPS 2026 paper
**ACED: Evaluating How LLM Agents Acquire and Act on Causal Evidence**.

ACED tests whether LLM agents can investigate a system: choose experiments, interpret finite evidence, and make causal
decisions. Each task pairs a natural-language scenario with a hidden causal model that defines the data, the effect of
every intervention, and the oracle answers. The benchmark has two suites:

- **ACED-Repair.** The agent diagnoses a failing production system described in a situation report (about 25
  controls and 26 measurable signals). With 15 experiments it decides which controls to test, reads confounded and
  filtered data, and recommends a fix with an account of the mechanism. The fix is re-simulated on the hidden true
  objective.
- **ACED-Query.** The agent answers causal questions and makes constrained causal decisions in worlds backed by
  categorical Bayesian networks.

## Quick start (about 10 minutes, no API calls)

Python 3.12. Clone the repository:

```bash
git clone https://github.com/Rose-STL-Lab/ACED-Bench.git
cd ACED-Bench
```

Then either install the pinned requirements (pgmpy needs torch; the CPU wheel is enough and much smaller, so
install it first):

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

or build the Docker image, which runs the tests by default:

```bash
docker build -t aced-bench .
docker run --rm aced-bench
```

Then:

```bash
bash prepare_dataset.sh       # the data ship with the repository; this verifies them against data/SHA256SUMS
bash run_tests.sh             # offline tests, about one minute
bash run_experiments.sh demo  # ACED-Repair end to end with a scripted agent, about one minute
```

The demo runs a scripted agent that measures, analyses its data, tests the oracle fix and submits the oracle answer on
the 54 evaluation worlds. Every world is accepted; the per-world results and `summary.json` are in `out/repair/demo/`.

## Running models

**Credentials.** Running models needs your own API access: the repository contains no keys, and the benchmark code
reads them from your environment and does not store them. Both suites map the agent's free-text requests to the hidden
model with a fixed parser model on AWS Bedrock (Claude Opus 4.8 for ACED-Repair, Claude Opus 4.6 for ACED-Query; the
ACED-Query answer extractor is Claude Opus 4.6 too), so AWS credentials are always needed: `AWS_BEARER_TOKEN_BEDROCK`
with `AWS_DEFAULT_REGION`, or the standard AWS credential chain (for an EC2/ECS instance role, also set
`BEDROCK_ALLOW_EC2_METADATA=1`; the region defaults to `us-west-2`). Replies are not streamed, so long replies need a
long read timeout: `BEDROCK_READ_TIMEOUT` defaults to 900 seconds. Agents can run on Bedrock, on any OpenAI-compatible
endpoint (`OPENAI_API_KEY`, `OPENAI_BASE_URL`), on a local vLLM server (`VLLM_BASE_URL`, ACED-Repair), or on Gemini
(`GEMINI_API_KEY`, ACED-Query).

### ACED-Repair

```bash
bash run_experiments.sh repair NAME BACKEND MODEL [SPLIT]     # SPLIT: eval (54 worlds, default) or test (540 worlds)

# examples
MAX_TOKENS=128000 bash run_experiments.sh repair opus55 bedrock us.anthropic.claude-opus-5-5
OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \
    bash run_experiments.sh repair mymodel openai <served model name> test
```

The script calls `src/aced_repair/run_benchmark.py`; `python src/aced_repair/run_benchmark.py --help` lists its
options and defaults. Rerunning the same command skips the worlds that already have a result (`--force` reruns them);
a world that was cut off restarts from its first turn, and a world that fails (e.g. an API error after retries) is
reported and left for the next run.

**Protocol.** 15 experiments and 32 turns per world. Each turn shows the agent the situation report, the result of its
last action and its own notes (`src/aced_repair/prompt.py` holds the agent prompt). Set `MAX_TOKENS` to the model's
maximum output length: the paper used 128000 for Claude Opus 5.5, Claude Sonnet 5.5, GPT-6 Astra and GPT-6.1 Sol, 8192
for Llama 4 Maverick, and 32768 for the other models.

**Ablations in the paper.** `--history full` sends the whole conversation every turn; `--budget 30` and `--budget 8`
change the experiment budget. `--no-resolver-llm` parses requests with the deterministic alias matcher alone (no
Bedrock calls for parsing).

**Sandbox.** The agent can run Python analysis code on its experiment data. The code runs in a separate process with a
cleaned environment, whose audit hook refuses file access outside its working folder, subprocesses, and ctypes file,
program and network calls. This is a safeguard against accidents, not a security boundary. World files contain the
answers, so for a public leaderboard run the benchmark in a container where `data/` is not mounted for the agent's
process.

**Output.** `out/repair/NAME/result_<world>.json` holds the full trajectory (every turn, what the agent saw, how each
request was parsed, code output) and the grade; `data_<hash>/` folders hold each world's experiment CSVs (hashed
names, because world ids name the causal structure); `summary.json` aggregates, with one row per world (`fix_ok`,
`mechanism_score`, `accepted`, ...). Grade fields:

| field | meaning |
|---|---|
| `benefit_recovered` | improvement of the true objective under the recommended fix, as a share of the oracle fix's improvement (negative for a harmful fix; the paper clips it at 0) |
| `part_a_utility_ok` | the fix works: `benefit_recovered >= 0.9` (the fix rate counts these worlds) |
| `battery_fraction` | mechanism score: the mechanism proxy, the confounded decoys, and the signs of the true levers |
| `accepted` | the fix works and the mechanism score is at least 0.8 |

In the grade, `recommended_intervention` and `gold_intervention` give each control's setting on an internal 0–100
scale of its range (the agent answers in the physical units of the situation report).

### ACED-Query

```bash
bash run_experiments.sh query NAME BACKEND MODEL

# example
bash run_experiments.sh query opus55 bedrock us.anthropic.claude-opus-5-5
```

This runs the paper's main protocol: the active reasoning, active-code and modular agents with a four-turn budget on
the structural (`core_struct`) and decision (`core_decision`) worlds, and the guess-only baseline without data
(`guess_struct`, `guess_decision`). It then scores every run and prints accuracy on the Qual and Core question sets.
Results go to `out/query/NAME/`; the header of `src/aced_query/run_model.sh` lists the options. As in the paper,
agents sample at temperature 0.3 and the guess-only baseline at 0.1 (`--temperature`); models whose API rejects a
temperature (Claude 5.5 and GPT-6 on Bedrock) run at the provider default.

**Code execution.** The active-code and modular agents run their analysis code in a separate Python process with no
file or network restrictions and with the caller's environment variables. Run untrusted models in a container that
holds no credentials beyond the API keys the run needs, and, for a public leaderboard, without `data/` mounted for the
agent's process (the world files contain the answers).

Run directly, `run_agent_batch.py` and `run_guess.py` write to `out/query/` and print the command that scores the
run; `--help` shows every option and its default.

For the sample-budget setting (at most 5,000 sampled rows per task and no turn limit; the cap of 10 data queries per
question still applies), call the runner directly, once per agent (`active`, `active_code`, `modular`):

```bash
python src/aced_query/run_agent_batch.py --worlds-dir data/query/core_decision --agent-type active_code \
    --limit-variant samples --max-total-samples 5000 \
    --scientist-backend bedrock --scientist-model <model> --max-new-tokens 65536 \
    --output-dir out/query/<name> -o out/query/<name>/<name>.json
python src/aced_query/evaluate_decision.py out/query/<name>/<name>.json -o out/query/<name>/eval.json
```

## Repository layout

```
src/aced_repair/      ACED-Repair: agent loop and prompt, simulator, request parser, oracle and grader, code sandbox
src/aced_query/       ACED-Query: agents, Bayesian-network simulator, request parser, guess-only baseline, scorers
data/repair/          test (540 worlds) and eval (the 54-world evaluation set, a subset of test)
data/query/           ACED-Query worlds with their questions
tests/                offline tests (no API calls)
prepare_dataset.sh    verifies the data files
run_tests.sh          runs the tests
run_experiments.sh    the demo, and a model on either suite
Dockerfile            the environment as a container
```

Pull requests run the checks in `.github/workflows`: flake8, the tests with a coverage report, a line count, and a
check that the three scripts above exist.

## License

The code and data are released under the MIT License (`LICENSE`).

## Contact

For questions, please open an issue or email Yiwei Chen (vic010@ucsd.edu).
