# ACED-Query

The agent answers causal questions and makes constrained causal decisions in worlds backed by categorical Bayesian
networks.

## Running a model

```bash
bash run_experiments.sh query NAME BACKEND MODEL

# example
bash run_experiments.sh query opus55 bedrock us.anthropic.claude-opus-5-5
```

This runs the paper's main protocol and scores it:

- the Active, Active+Code and Modular agents with a four-turn budget on the structural (`core_struct`) and decision
  (`core_decision`) worlds;
- the guess-only baseline without data (`guess_struct`, `guess_decision`).

It prints accuracy on the Qual and Core question sets. Results go to `out/query/NAME/`; the header of
`src/aced_query/run_model.sh` lists the options. Agents sample at temperature 0.3 and the guess-only baseline at 0.1
(`--temperature`); models whose API rejects a temperature (Claude 5.5 and GPT-6 on Bedrock) run at the provider
default.

Run directly, `src/aced_query/run_agent_batch.py` and `src/aced_query/run_guess.py` write to `out/query/` and print the
command that scores the run; `--help` shows every option and its default.

## Sample-budget setting

At most 5,000 sampled rows per task and no turn limit (the cap of 10 data queries per question still applies). Run it
once per agent (`active`, `active_code`, `modular`):

```bash
python src/aced_query/run_agent_batch.py --worlds-dir data/query/core_decision --agent-type active_code \
    --limit-variant samples --max-total-samples 5000 \
    --scientist-backend bedrock --scientist-model <model> --max-new-tokens 65536 \
    --output-dir out/query/<name> -o out/query/<name>/<name>.json
python src/aced_query/evaluate_decision.py out/query/<name>/<name>.json -o out/query/<name>/eval.json
```
