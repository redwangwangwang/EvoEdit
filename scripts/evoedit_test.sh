#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python || true)"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "No usable Python interpreter found; set PYTHON_BIN explicitly." >&2
  exit 1
fi

: "${CKPT_FILE:?Set CKPT_FILE to an EvoEdit Lightning checkpoint.}"
DATASET=${DATASET:-longitudinal-mimic}
ANNOTATION=${ANNOTATION:-./dataset/mimic-cxr/annotation.json}
BASE_DIR=${BASE_DIR:-./dataset/mimic-cxr/}
VISION_MODEL=${VISION_MODEL:-./pretrain_weights/swin-base-patch4-window7-224/}
LLAMA_MODEL=${LLAMA_MODEL:-./pretrain_weights/Llama-2-7b-chat-hf/}
SAVE_PATH=${SAVE_PATH:-./save/${DATASET}/evoedit-test}
ALLOW_MISSING_PROGRESSIONS=${ALLOW_MISSING_PROGRESSIONS:-False}

mkdir -p "${SAVE_PATH}"

"${PYTHON_BIN}" -u train.py \
  --stage evoedit \
  --test \
  --ckpt_file "${CKPT_FILE}" \
  --dataset "${DATASET}" \
  --annotation "${ANNOTATION}" \
  --base_dir "${BASE_DIR}" \
  --vision_model "${VISION_MODEL}" \
  --llama_model "${LLAMA_MODEL}" \
  --savedmodel_path "${SAVE_PATH}" \
  --devices "${DEVICES:-1}" \
  --test_batch_size "${TEST_BATCH_SIZE:-4}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --strategy "${STRATEGY:-auto}" \
  --precision "${PRECISION:-bf16-mixed}" \
  --allow_missing_progressions "${ALLOW_MISSING_PROGRESSIONS}" \
  2>&1 | tee -a "${SAVE_PATH}/test.log"
