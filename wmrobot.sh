#!/usr/bin/env bash

set -euo pipefail

if [[ "${1:-}" == "--all-maps" ]]; then
    export WMROBOT_ALL_MAPS=1
    shift
fi
if (( $# != 0 )); then
    echo "Usage: $0 [--all-maps]" >&2
    exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/venv/bin/python3"
MATPLOTLIB_CONFIG_DIR="/tmp/gr-mppi-wmrobot-matplotlib"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 1
fi

mkdir -p "${MATPLOTLIB_CONFIG_DIR}"
cd "${PROJECT_ROOT}"

examples=(
    "src/wmrobot/example_mppi.py"
    "src/wmrobot/example_logmppi.py"
    "src/wmrobot/example_grmppi_v2.py"
)

for example in "${examples[@]}"; do
    echo ""
    echo "===== Running ${example} ====="
    MPLBACKEND=Agg MPLCONFIGDIR="${MATPLOTLIB_CONFIG_DIR}" \
        "${PYTHON_BIN}" "${PROJECT_ROOT}/${example}"
done
