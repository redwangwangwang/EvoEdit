# EvoEdit

**EvoEdit: Executable Clinical Edit Programs for Longitudinal Radiology Report Generation**

EvoEdit is a model-level extension of [TIM](https://github.com/yihengd/TIM). It keeps TIM's original longitudinal MIMIC-CXR pairing, annotation file, and train/validation/test splits unchanged. Instead of compressing temporal evolution into one holistic progression embedding, EvoEdit predicts a sparse finding-wise edit program and executes it against the prior report.

> **Do not regenerate what has not changed.**

## Method

For each of the 13 CheXbert findings, EvoEdit predicts one operation: `KEEP`, `APPEAR`, `RESOLVE`, `WORSEN`, `IMPROVE`, or `UNCERTAIN`.

The implementation adds:

1. **Soft temporal correspondence** to align prior and current visual tokens before change modeling.
2. **Factorized edit slots** containing finding, operation, latent anatomy, latent severity, and confidence.
3. **Copy-and-edit execution** that retrieves finding-specific facts from the prior report and uses `KEEP` as an explicit preservation gate.
4. **Invertible edit algebra** enforcing `APPEAR ↔ RESOLVE` and `WORSEN ↔ IMPROVE`.
5. **Intervention verification** that suppresses one edit and checks that the remaining program stays unchanged.
6. **Pointer-copy supervision** for stable report tokens while retaining standard Hugging Face generation.

Operation targets are created online from the two reports already present in every TIM sample using frozen CheXbert states and sentence-level direction phrases. No image, annotation, split, registration target, or transition label is added or rewritten.

## Layout

```text
TIM/                               pinned upstream TIM submodule
models/model_evoedit.py            assembled Lightning model
models/evoedit/model_core.py       TIM integration and edit execution
models/evoedit/model_training.py   training losses and bidirectional program
models/evoedit/model_generation.py generation and JSON audit trail
models/evoedit/program.py          correspondence, tokenizer, executor
models/evoedit/targets.py          online operation targets
models/evoedit/losses.py           inverse/cycle/sparse/verifier losses
models/evoedit/copy.py             pointer-copy objective
configs/config.py                  TIM-compatible EvoEdit arguments
scripts/evoedit_train.sh           training entry point
scripts/evoedit_test.sh            evaluation entry point
```

The original TIM stages remain available through symbolic links and can be selected with `--stage stage1` or `--stage stage2`.

## Clone

The TIM submodule is required:

```bash
git clone --recurse-submodules https://github.com/redwangwangwang/EvoEdit.git
cd EvoEdit
```

For an existing clone:

```bash
git submodule update --init --recursive
```

## Installation

```bash
conda create -n evoedit python=3.9
conda activate evoedit
pip install -r requirements.txt
```

Use the same pretrained weights as TIM under `pretrain_weights/`: BERT-base, CheXbert, Llama-2-7B-chat, Swin-base, and XCLIP-base. Prepare MIMIC-CXR exactly as described by TIM; EvoEdit does not modify the annotation JSON.

## Train

```bash
bash scripts/evoedit_train.sh
```

Example with custom paths:

```bash
ANNOTATION=/data/mimic/annotation.json \
BASE_DIR=/data/mimic-cxr \
VISION_MODEL=/weights/swin-base-patch4-window7-224 \
LLAMA_MODEL=/weights/Llama-2-7b-chat-hf \
DEVICES=2 \
bash scripts/evoedit_train.sh
```

## Test

```bash
CKPT_FILE=/path/to/checkpoint.ckpt bash scripts/evoedit_test.sh
```

EvoEdit writes generated reports and an auditable finding-wise edit program under `<savedmodel_path>/result/`. Each entry includes the operation, anatomy/severity code, confidence, and preservation gate.

## Checks

```bash
python -m py_compile train.py configs/config.py models/model_evoedit.py models/evoedit/*.py
pytest -q tests/test_evoedit_core.py
bash -n scripts/evoedit_train.sh
bash -n scripts/evoedit_test.sh
```

Full training requires the restricted MIMIC-CXR data and TIM's pretrained weights.

## Acknowledgement

This repository is built on TIM and inherits TIM's acknowledgement of R2GenGPT. Please cite the original projects when using this code.
