# ACED-Query

## Running a model

```bash
bash run_experiments.sh query NAME BACKEND MODEL

# example
bash run_experiments.sh query opus55 bedrock us.anthropic.claude-opus-5-5
```

This runs the paper's main protocol (the Active, Active+Code and Modular agents with four turns, and the guess-only
baseline), scores it, and prints accuracy on the Qual and Core sets. Results go to `out/query/NAME/`; the header of
`src/aced_query/run_model.sh` lists the options.

## Sample-budget setting

At most 5,000 sampled rows per task instead of four turns. Run it once per agent (`active`, `active_code`, `modular`):

```bash
python src/aced_query/run_agent_batch.py --worlds-dir data/query/core_decision --agent-type active_code \
    --limit-variant samples --max-total-samples 5000 \
    --scientist-backend bedrock --scientist-model <model> --max-new-tokens 65536 \
    --output-dir out/query/<name> -o out/query/<name>/<name>.json
python src/aced_query/evaluate_decision.py out/query/<name>/<name>.json -o out/query/<name>/eval.json
```
