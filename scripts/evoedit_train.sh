#!/usr/bin/env bash
set -euo pipefail

DATASET=${DATASET:-longitudinal-mimic}
ANNOTATION=${ANNOTATION:-./dataset/mimic-cxr/annotation.json}
BASE_DIR=${BASE_DIR:-./dataset/mimic-cxr/}
VISION_MODEL=${VISION_MODEL:-./pretrain_weights/swin-base-patch4-window7-224/}
LLAMA_MODEL=${LLAMA_MODEL:-./pretrain_weights/Llama-2-7b-chat-hf/}
SAVE_PATH=${SAVE_PATH:-./save/${DATASET}/evoedit}
DEVICES=${DEVICES:-2}

mkdir -p "${SAVE_PATH}"

python -u train.py \
  --stage evoedit \
  --dataset "${DATASET}" \
  --annotation "${ANNOTATION}" \
  --base_dir "${BASE_DIR}" \
  --vision_model "${VISION_MODEL}" \
  --llama_model "${LLAMA_MODEL}" \
  --savedmodel_path "${SAVE_PATH}" \
  --batch_size "${BATCH_SIZE:-2}" \
  --val_batch_size "${VAL_BATCH_SIZE:-4}" \
  --devices "${DEVICES}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --max_epochs "${MAX_EPOCHS:-3}" \
  --accumulate_grad_batches "${ACCUMULATE_GRAD_BATCHES:-2}" \
  --freeze_vm "${FREEZE_VM:-False}" \
  --vis_use_lora "${VIS_USE_LORA:-False}" \
  --llm_use_lora "${LLM_USE_LORA:-False}" \
  --strategy "${STRATEGY:-deepspeed}" \
  --precision "${PRECISION:-bf16-mixed}" \
  2>&1 | tee -a "${SAVE_PATH}/train.log"
