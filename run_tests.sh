#!/usr/bin/env bash
# Offline tests (no API calls, about one minute): data integrity, the ACED-Repair sandbox, an end-to-end ACED-Repair run
# with a scripted agent that must be accepted, and ACED-Query world simulation.
#
#   bash run_tests.sh
set -euo pipefail
cd "$(dirname "$0")"
python -m pytest -q tests "$@"
