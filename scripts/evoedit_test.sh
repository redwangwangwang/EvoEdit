#!/usr/bin/env bash
set -euo pipefail

: "${CKPT_FILE:?Set CKPT_FILE to an EvoEdit Lightning checkpoint.}"
DATASET=${DATASET:-longitudinal-mimic}
ANNOTATION=${ANNOTATION:-./dataset/mimic-cxr/annotation.json}
BASE_DIR=${BASE_DIR:-./dataset/mimic-cxr/}
VISION_MODEL=${VISION_MODEL:-./pretrain_weights/swin-base-patch4-window7-224/}
LLAMA_MODEL=${LLAMA_MODEL:-./pretrain_weights/Llama-2-7b-chat-hf/}
SAVE_PATH=${SAVE_PATH:-./save/${DATASET}/evoedit-test}

mkdir -p "${SAVE_PATH}"

python -u train.py \
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
  2>&1 | tee -a "${SAVE_PATH}/test.log"
