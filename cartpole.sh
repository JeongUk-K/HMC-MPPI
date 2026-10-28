#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/venv/bin/python"
MATPLOTLIB_CONFIG_DIR="/tmp/cr-mppi-matplotlib"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 1
fi

mkdir -p "${MATPLOTLIB_CONFIG_DIR}"
cd "${PROJECT_ROOT}"

examples=(
    "src/cartpole/example_mppi_sigma5.py"
    "src/cartpole/example_mppi_sigma10.py"
    "src/cartpole/example_mppi_sigma15.py"
    "src/cartpole/example_grmppi.py"
    "src/cartpole/example_grmppi_v2.py"
    "src/cartpole/example_logmppi.py"
)

for example in "${examples[@]}"; do
    echo ""
    echo "===== Running ${example} ====="
    MPLBACKEND=Agg MPLCONFIGDIR="${MATPLOTLIB_CONFIG_DIR}" \
        "${PYTHON_BIN}" "${PROJECT_ROOT}/${example}"
done

echo ""
echo "===== All 6 CartPole examples completed ====="
