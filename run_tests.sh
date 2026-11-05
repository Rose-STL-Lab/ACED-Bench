#!/usr/bin/env bash
# Offline tests (no API calls, about one minute).
#
#   bash run_tests.sh
set -euo pipefail
cd "$(dirname "$0")"
python -m pytest -q tests "$@"
