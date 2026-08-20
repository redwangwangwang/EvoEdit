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

DATASET=${DATASET:-longitudinal-mimic}
ANNOTATION=${ANNOTATION:-./dataset/mimic-cxr/annotation.json}
BASE_DIR=${BASE_DIR:-./dataset/mimic-cxr/}
VISION_MODEL=${VISION_MODEL:-./pretrain_weights/swin-base-patch4-window7-224/}
LLAMA_MODEL=${LLAMA_MODEL:-./pretrain_weights/Llama-2-7b-chat-hf/}
SAVE_PATH=${SAVE_PATH:-./save/${DATASET}/evoedit}
ALLOW_MISSING_PROGRESSIONS=${ALLOW_MISSING_PROGRESSIONS:-False}

mkdir -p "${SAVE_PATH}"

"${PYTHON_BIN}" -u train.py \
  --stage evoedit \
  --dataset "${DATASET}" \
  --annotation "${ANNOTATION}" \
  --base_dir "${BASE_DIR}" \
  --vision_model "${VISION_MODEL}" \
  --llama_model "${LLAMA_MODEL}" \
  --savedmodel_path "${SAVE_PATH}" \
  --batch_size "${BATCH_SIZE:-2}" \
  --val_batch_size "${VAL_BATCH_SIZE:-4}" \
  --freeze_vm "${FREEZE_VM:-False}" \
  --vis_use_lora "${VIS_USE_LORA:-False}" \
  --llm_use_lora "${LLM_USE_LORA:-False}" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --gradient_clip_val "${GRADIENT_CLIP_VAL:-1}" \
  --max_length "${MAX_LENGTH:-150}" \
  --min_new_tokens "${MIN_NEW_TOKENS:-80}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-150}" \
  --repetition_penalty "${REPETITION_PENALTY:-2.0}" \
  --length_penalty "${LENGTH_PENALTY:-2.0}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --devices "${DEVICES:-2}" \
  --max_epochs "${MAX_EPOCHS:-3}" \
  --limit_val_batches "${LIMIT_VAL_BATCHES:-1.0}" \
  --val_check_interval "${VAL_CHECK_INTERVAL:-0.5}" \
  --num_sanity_val_steps "${NUM_SANITY_VAL_STEPS:-2}" \
  --accumulate_grad_batches "${ACCUMULATE_GRAD_BATCHES:-2}" \
  --strategy "${STRATEGY:-deepspeed}" \
  --precision "${PRECISION:-bf16-mixed}" \
  --allow_missing_progressions "${ALLOW_MISSING_PROGRESSIONS}" \
  "$@" \
  2>&1 | tee -a "${SAVE_PATH}/train.log"
