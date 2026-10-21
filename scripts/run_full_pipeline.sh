#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATIENT_ID="${1:-563}"
DEVICE="${2:-gpu}"
RUN_NAME="compact_l32_d128_d64"
RUN_ROOT="${ROOT_DIR}/outputs/compact_qat_patient_${PATIENT_ID}"
RUN_DIR="${RUN_ROOT}/runs/${RUN_NAME}"
TEST_XML="${ROOT_DIR}/data/OhioT1DM/2018/test/${PATIENT_ID}-ws-testing.xml"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[run_full_pipeline] missing virtual environment at ${ROOT_DIR}/.venv" >&2
  echo "[run_full_pipeline] run ./scripts/bootstrap.sh first" >&2
  exit 1
fi

if [[ ! -f "${TEST_XML}" ]]; then
  echo "[run_full_pipeline] test XML not found: ${TEST_XML}" >&2
  exit 1
fi

echo "[run_full_pipeline] training fixed-QAT model for patient ${PATIENT_ID}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/train_qat.py" --patient-id "${PATIENT_ID}" --device "${DEVICE}"

echo "[run_full_pipeline] evaluating trained checkpoint"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/evaluate_qat.py" \
  --run-dir "${RUN_DIR}" \
  --test-xml "${TEST_XML}" \
  --device "${DEVICE}"

echo "[run_full_pipeline] completed for patient ${PATIENT_ID} on device ${DEVICE}"
