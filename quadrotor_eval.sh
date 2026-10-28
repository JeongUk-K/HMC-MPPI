#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/venv/bin/python"
MATPLOTLIB_CONFIG_DIR="/tmp/cr-mppi-quadrotor-matplotlib"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 1
fi

mkdir -p "${MATPLOTLIB_CONFIG_DIR}"
cd "${PROJECT_ROOT}"

for example in \
    src/quadrotor_eval/example_mppi.py \
    src/quadrotor_eval/example_grmppi_v2.py \
    src/quadrotor_eval/example_logmppi.py; do
    echo "===== Running ${example} ====="
    MPLBACKEND=Agg MPLCONFIGDIR="${MATPLOTLIB_CONFIG_DIR}" \
        "${PYTHON_BIN}" "${PROJECT_ROOT}/${example}"
done

echo "===== Comparing quadrotor results and rendering MP4 ====="
MPLBACKEND=Agg MPLCONFIGDIR="${MATPLOTLIB_CONFIG_DIR}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/src/quadrotor_eval/compare_results.py"
