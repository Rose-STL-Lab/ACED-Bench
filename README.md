# ACED: Active Causal Evidence and Decision Benchmark

Code and data for **ACED: Evaluating How LLM Agents Acquire and Act on Causal Evidence** (NeurIPS 2026).

ACED tests whether LLM agents can investigate a system: choose experiments, interpret finite evidence, and make causal
decisions. Each task pairs a natural-language scenario with a hidden causal model that defines the data, the effect of
every intervention, and the oracle answers.

- **ACED-Repair**: diagnose a failing production system within 15 experiments and recommend a fix, which is
  re-simulated on the hidden true objective.
- **ACED-Query**: answer causal questions and make constrained causal decisions in worlds backed by categorical
  Bayesian networks.

## Installation

```bash
git clone https://github.com/Rose-STL-Lab/ACED-Bench.git
cd ACED-Bench
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Python 3.12. A Docker image is also provided ([docs/setup.md](docs/setup.md)).

## Quick start (no API calls)

```bash
bash prepare_dataset.sh       # verify the data
bash run_tests.sh             # offline tests, about one minute
bash run_experiments.sh demo  # a scripted agent on ACED-Repair, about one minute
```

## Evaluating a model

Use your own API keys. The request parser runs on AWS Bedrock, so AWS credentials are always needed
([docs/setup.md](docs/setup.md)).

```bash
MAX_TOKENS=128000 bash run_experiments.sh repair opus55 bedrock us.anthropic.claude-opus-5-5
bash run_experiments.sh query opus55 bedrock us.anthropic.claude-opus-5-5
```

Results go to `out/repair/opus55/` and `out/query/opus55/`. Protocols, options, outputs and the paper's ablations are
in [docs/aced_repair.md](docs/aced_repair.md) and [docs/aced_query.md](docs/aced_query.md). Agents run their analysis
code on your machine; read [docs/setup.md](docs/setup.md) before evaluating untrusted models.

## Repository layout

```
src/aced_repair/   ACED-Repair: agent loop, simulator, request parser, oracle and grader, code sandbox
src/aced_query/    ACED-Query: agents, Bayesian-network simulator, request parser, scorers
data/              ACED-Repair worlds (540 test, 54 eval) and ACED-Query worlds
tests/             offline tests
docs/              setup and per-suite documentation
```

## License

MIT (see [LICENSE](LICENSE)).

## Contact

Please open an issue or email Yiwei Chen (vic010@ucsd.edu).
