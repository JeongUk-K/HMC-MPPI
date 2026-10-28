#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/venv/bin/python3"
MATPLOTLIB_CONFIG_DIR="/tmp/gr-mppi-wmrobot-crowd-matplotlib"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 1
fi

mkdir -p "${MATPLOTLIB_CONFIG_DIR}"
cd "${PROJECT_ROOT}"

examples=(
    "src/wmrobot_crowd/example_mppi.py"
    "src/wmrobot_crowd/example_logmppi.py"
    "src/wmrobot_crowd/example_grmppi_v2.py"
)

for example in "${examples[@]}"; do
    echo ""
    echo "===== Running ${example} ====="
    MPLBACKEND=Agg MPLCONFIGDIR="${MATPLOTLIB_CONFIG_DIR}" \
        "${PYTHON_BIN}" "${PROJECT_ROOT}/${example}"
done

echo ""
echo "===== Comparing moving-crowd results ====="
MPLBACKEND=Agg MPLCONFIGDIR="${MATPLOTLIB_CONFIG_DIR}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/src/wmrobot_crowd/compare_results.py"
