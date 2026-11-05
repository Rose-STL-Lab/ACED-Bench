#!/usr/bin/env bash
# Run one model on ACED-Query with the paper's main protocol, then score it.
#
#   bash run_model.sh NAME BACKEND MODEL
#
#   NAME     short tag used for folders and run names (e.g. opus55)
#   BACKEND  bedrock | openai | gemini
#   MODEL    model id for that backend
#
# What it runs:
#   agents  4-turn budget, 3 agents (active, active_code = Active+Code, modular)
#           x 2 sets (core_struct: 10 worlds / 60 questions; core_decision: 60 worlds / 60 questions)
#   guess   guess-only, no data (guess_struct: 30 worlds / 180 questions; guess_decision: 48 worlds / 48 questions)
#   eval    score everything into <out>/eval/ and print accuracy on the Qual and Core sets
#
# Environment:
#   BASE_URL, API_KEY  for BACKEND=openai (vLLM, DeepSeek, ...); otherwise OPENAI_BASE_URL / OPENAI_API_KEY
#   MAX_TOKENS         agent output cap, default 65536 (set it to the model's maximum output length)
#   GUESS_MAX_TOKENS   guess-only output cap, default = MAX_TOKENS
#   STAGES             default "run eval"; "eval" alone re-scores finished runs
#   STRUCT_SHARDS / DECISION_SHARDS   parallel workers per agent and set, default 10 / 20
#   OUT_ROOT           default: <repo>/out/query
#   PYTHON             default: python
#
# Safe to stop and restart: finished questions are skipped (agent_logs/ in each run folder).
set -uo pipefail

if [ $# -lt 3 ]; then sed -n 2,25p "$0"; exit 1; fi
NAME=$1; BACKEND=$2; MODEL=$3
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
DATA=$ROOT/data/query
OUT=${OUT_ROOT:-$ROOT/out/query}/$NAME
PY=${PYTHON:-python}
MAX_TOKENS=${MAX_TOKENS:-65536}
GUESS_MAX_TOKENS=${GUESS_MAX_TOKENS:-$MAX_TOKENS}
STAGES=${STAGES:-"run eval"}
STRUCT_SHARDS=${STRUCT_SHARDS:-10}
DECISION_SHARDS=${DECISION_SHARDS:-20}
# Request parser and answer extractor (the same for every agent model)
PARSER=us.anthropic.claude-opus-4-6-v1
export BEDROCK_READ_TIMEOUT=${BEDROCK_READ_TIMEOUT:-900}

SCI=(--scientist-backend "$BACKEND" --scientist-model "$MODEL")
GUESS=(--backend "$BACKEND" --model "$MODEL")
if [ -n "${BASE_URL:-}" ]; then SCI+=(--scientist-base-url "$BASE_URL"); GUESS+=(--base-url "$BASE_URL"); fi
if [ -n "${API_KEY:-}" ]; then SCI+=(--scientist-api-key "$API_KEY"); GUESS+=(--api-key "$API_KEY"); fi

mkdir -p "$OUT/logs" "$OUT/eval" "$OUT/guess"

# One cell = one agent on one world set. Shards run in parallel into the same run folder; the final pass over the full
# set only collects the finished questions (and redoes any a shard missed) into <run>.json.
run_cell() {
  local set=$1 agent=$2 nshard=$3
  local run=${NAME}_${agent}_${set}
  local dir=$OUT/agents/$run
  local common=(--run-name "$run" --output-dir "$dir" --agent-type "$agent"
                --limit-variant rounds --max-turns 4
                --max-new-tokens "$MAX_TOKENS" --world-model "$PARSER" "${SCI[@]}")
  mkdir -p "$dir"
  local pids=() i
  for ((i = 0; i < nshard; i++)); do
    local sd=$OUT/shards/$set/$i
    ls "$sd"/world_*.json >/dev/null 2>&1 || continue
    mkdir -p "$OUT/tmp/${run}_$i"
    TMPDIR=$OUT/tmp/${run}_$i $PY "$HERE/run_agent_batch.py" --worlds-dir "$sd" -o "$dir/shard_$i.json" \
      "${common[@]}" > "$OUT/logs/${run}_shard_$i.log" 2>&1 &
    pids+=($!)
    sleep 1
  done
  wait "${pids[@]}"
  mkdir -p "$OUT/tmp/${run}_merge"
  TMPDIR=$OUT/tmp/${run}_merge $PY "$HERE/run_agent_batch.py" --worlds-dir "$DATA/$set" -o "$dir/$run.json" \
    "${common[@]}" > "$OUT/logs/${run}_merge.log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M)] done $run (exit $rc)"
}

make_shards() {  # set nshard: world i goes to shard i mod nshard (symlinks); made once, before the cells start
  local set=$1 nshard=$2 worlds=("$DATA/$1"/world_*.json) i
  [ -d "$OUT/shards/$set" ] && return
  for ((i = 0; i < ${#worlds[@]}; i++)); do
    mkdir -p "$OUT/shards/$set/$((i % nshard))"
    ln -s "${worlds[$i]}" "$OUT/shards/$set/$((i % nshard))/"
  done
}

run_guess() {  # writes guess_<set>_<difficulty>.json (one file per difficulty group)
  local set=$1
  if ls "$OUT/guess/${set}"_*.json >/dev/null 2>&1; then echo "skip guess $set (exists)"; return; fi
  $PY "$HERE/run_guess.py" --worlds-dir "$DATA/$set" "${GUESS[@]}" --max-new-tokens "$GUESS_MAX_TOKENS" \
    -o "$OUT/guess/${set}.json" > "$OUT/logs/${set}.log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M)] done $set (exit $rc)"
}

score() {  # evaluator, output, inputs...
  local ev=$1 o=$2; shift 2
  for f in "$@"; do [ -f "$f" ] || { echo "MISSING $f" >&2; return 1; }; done
  $PY "$HERE/$ev" "$@" --details --extract-model "$PARSER" -o "$o" > "$o.log" 2>&1 \
    && echo "scored $(basename "$o")" || echo "FAILED $(basename "$o") (see $o.log)" >&2
}

if [[ " $STAGES " == *" run "* ]]; then
  echo "[$(date +%H:%M)] $NAME: $BACKEND $MODEL -> $OUT"
  make_shards core_struct "$STRUCT_SHARDS"
  make_shards core_decision "$DECISION_SHARDS"
  pids=()
  for agent in active active_code modular; do
    run_cell core_struct "$agent" "$STRUCT_SHARDS" & pids+=($!)
    run_cell core_decision "$agent" "$DECISION_SHARDS" & pids+=($!)
  done
  run_guess guess_struct & pids+=($!)
  run_guess guess_decision & pids+=($!)
  wait "${pids[@]}"
fi

if [[ " $STAGES " == *" eval "* ]]; then
  for agent in active active_code modular; do
    r=${NAME}_${agent}_core_struct
    score evaluate_struct.py "$OUT/eval/eval_$r.json" "$OUT/agents/$r/$r.json"
    r=${NAME}_${agent}_core_decision
    score evaluate_decision.py "$OUT/eval/eval_$r.json" "$OUT/agents/$r/$r.json"
  done
  score evaluate_struct.py "$OUT/eval/eval_${NAME}_guess_struct.json" \
    "$OUT/guess/guess_struct_easy.json" "$OUT/guess/guess_struct_medium.json"
  score evaluate_decision.py "$OUT/eval/eval_${NAME}_guess_decision.json" "$OUT/guess/guess_decision_hard.json"
  $PY "$HERE/summarize.py" "$OUT"
fi
