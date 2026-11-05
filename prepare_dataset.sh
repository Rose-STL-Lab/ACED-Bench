#!/usr/bin/env bash
# The benchmark worlds ship with the repository (data/, about 11 MB), so there is nothing to download. This script
# checks that they are complete and unmodified.
#
#   bash prepare_dataset.sh
#
#   data/repair/test           540 ACED-Repair test worlds, 60 per causal structure (9 structures) across 10 domains
#                              (clinical and fermentation about 135 worlds each, the other eight 31-37)
#   data/repair/eval            54 of the test worlds (6 per structure), the evaluation set used for the multi-model runs
#   data/query/core_struct      10 ACED-Query worlds, 60 structural questions
#   data/query/core_decision    60 ACED-Query worlds, 60 causal-decision questions
#   data/query/guess_struct     30 worlds, 180 structural questions for the guess-only baseline
#   data/query/guess_decision   48 worlds, 48 causal-decision questions for the guess-only baseline
set -euo pipefail
cd "$(dirname "$0")/data"
sha256sum --quiet -c SHA256SUMS && echo "data OK: $(grep -c . SHA256SUMS) files verified"
