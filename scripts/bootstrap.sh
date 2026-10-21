#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${ROOT_DIR}/.venv"

echo "[bootstrap] root=${ROOT_DIR}"
"${PYTHON_BIN}" --version

"${PYTHON_BIN}" - <<'PY'
import sys
major, minor = sys.version_info[:2]
if not (major == 3 and minor in (8, 9)):
    raise SystemExit(
        "This repository is tested with Python 3.8/3.9. "
        f"Detected Python {major}.{minor}. "
        "Set PYTHON_BIN to a Python 3.8/3.9 interpreter and rerun."
    )
PY

if [[ -d "${VENV_DIR}" ]]; then
  echo "[bootstrap] reusing existing virtual environment at ${VENV_DIR}"
else
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/requirements.txt"

echo "[bootstrap] environment ready at ${VENV_DIR}"
echo "[bootstrap] activate with: source ${VENV_DIR}/bin/activate"
echo "[bootstrap] one-command pipeline: ${ROOT_DIR}/scripts/run_full_pipeline.sh 563"
