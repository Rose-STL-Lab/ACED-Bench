#!/usr/bin/env bash
# Run a model on ACED-Repair or ACED-Query with the paper's protocol.
#
#   bash run_experiments.sh demo                                # ACED-Repair with a scripted agent, no API (~1 min)
#   bash run_experiments.sh repair NAME BACKEND MODEL [SPLIT]   # SPLIT: eval (54 worlds, default) or test (540)
#   bash run_experiments.sh query  NAME BACKEND MODEL
#
#   NAME     short tag for the output folder, e.g. opus55
#   BACKEND  bedrock | openai | vllm (repair) ; bedrock | openai | gemini (query)
#   MODEL    model id for that backend
#
# Results go to out/repair/NAME or out/query/NAME. Runs are safe to stop and restart.
# Environment (see docs/setup.md): AWS credentials for Bedrock (always needed: the request parser runs there),
# OPENAI_API_KEY / OPENAI_BASE_URL (or VLLM_BASE_URL) for OpenAI-compatible endpoints, GEMINI_API_KEY for Gemini,
# MAX_TOKENS (output cap; default 32768 for repair, 65536 for query), CONCURRENCY (repair, default 8).
set -euo pipefail
cd "$(dirname "$0")"
MODE=${1:-}; shift || true
case "$MODE" in
  demo)
    python src/aced_repair/run_benchmark.py --worlds-dir data/repair/eval --backend mock --outdir out/repair/demo
    ;;
  repair)
    [ $# -ge 3 ] || { sed -n 2,15p "$0"; exit 1; }
    NAME=$1 BACKEND=$2 MODEL=$3 SPLIT=${4:-eval}
    python src/aced_repair/run_benchmark.py --worlds-dir "data/repair/$SPLIT" --backend "$BACKEND" --model "$MODEL" \
      --max-new-tokens "${MAX_TOKENS:-32768}" --concurrency "${CONCURRENCY:-8}" --outdir "out/repair/$NAME"
    ;;
  query)
    [ $# -ge 3 ] || { sed -n 2,15p "$0"; exit 1; }
    MAX_TOKENS=${MAX_TOKENS:-65536} bash src/aced_query/run_model.sh "$@"
    ;;
  *) sed -n 2,15p "$0"; exit 1 ;;
esac
