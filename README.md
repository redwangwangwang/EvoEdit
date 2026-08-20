# EvoEdit

**EvoEdit: Executable Clinical Edit Programs for Longitudinal Radiology Report Generation**

EvoEdit is a model-level extension of [TIM](https://github.com/yihengd/TIM). It keeps TIM's longitudinal MIMIC-CXR pairing, report preprocessing, train/validation/test splits, optimizer, training schedule, and Stage-I decoding policy unchanged. The method change is confined to replacing TIM's holistic progression representation with a finding-wise executable clinical edit program.

> **Do not regenerate what has not changed.**

## Method

For each of the 13 CheXbert findings, EvoEdit predicts one operation: `KEEP`, `APPEAR`, `RESOLVE`, `WORSEN`, `IMPROVE`, or `UNCERTAIN`.

The implementation adds:

1. **Soft temporal correspondence** to align prior and current visual tokens before change modeling.
2. **Factorized edit slots** containing finding, operation, latent anatomy, latent severity, and confidence.
3. **Copy-and-edit execution** that retrieves finding-specific facts from the prior report.
4. **Invertible edit algebra** enforcing `APPEAR ↔ RESOLVE` and `WORSEN ↔ IMPROVE`.
5. **Intervention verification** that suppresses one active edit and verifies the remaining active program.
6. **Pointer-copy supervision** for stable report tokens while retaining TIM's standard Hugging Face generation.

Operation targets are created online from the two reports already present in every TIM sample using frozen CheXbert states and finding-local direction phrases. EvoEdit does not require a rewritten annotation file. Existing `progressions`/`Changes` fields may remain in an annotation JSON, but the current EvoEdit operation targets and report generator do not consume those fields.

## Stability fixes

The edit branch is protected against the all-`KEEP` shortcut by:

- class-balanced focal operation supervision;
- conservative target construction that does not treat an omitted finding as resolved;
- active-slot weighting for inverse and cycle consistency;
- explicit preserve-gate supervision, with the direct `KEEP → gate` gradient detached;
- confidence calibrated to operation correctness rather than merely `UNCERTAIN` labels;
- edit-rate calibration instead of a loss that always rewards `KEEP`;
- sharp-and-balanced latent factor usage;
- collapse-sensitive logs such as `target_non_keep_rate`, `pred_non_keep_rate`, `non_keep_f1`, `mean_keep_probability`, and `mean_preserve_gate`.

The prompt template and decoding call follow TIM Stage I. In particular, the official launch defaults remain:

```text
max_length=150
min_new_tokens=80
max_new_tokens=150
beam_size=3
repetition_penalty=2.0
length_penalty=2.0
```

## Layout

```text
TIM/                               pinned upstream TIM submodule
models/model_evoedit.py            assembled Lightning model
models/evoedit/model_core.py       TIM integration and edit execution
models/evoedit/model_training.py   training losses and diagnostics
models/evoedit/model_generation.py TIM-aligned generation and JSON audit trail
models/evoedit/program.py          correspondence, edit slots, executor
models/evoedit/targets.py          conservative online operation targets
models/evoedit/losses.py           balanced/active edit objectives
models/evoedit/copy.py             pointer-copy objective
tools/audit_annotation.py          read-only annotation/style audit
configs/config.py                  TIM-aligned shared settings plus method options
scripts/evoedit_train.sh           training entry point
scripts/evoedit_test.sh            evaluation entry point
```

The original TIM stages remain available through symbolic links and can be selected with `--stage stage1` or `--stage stage2`.

## Clone and install

```bash
git clone --recurse-submodules https://github.com/redwangwangwang/EvoEdit.git
cd EvoEdit
conda create -n evoedit python=3.9
conda activate evoedit
pip install -r requirements.txt
```

For an existing clone:

```bash
git submodule update --init --recursive
```

Use the same pretrained weights as TIM under `pretrain_weights/`: BERT-base, CheXbert, Llama-2-7B-chat, Swin-base, and XCLIP-base.

## Audit an annotation without modifying it

```bash
python tools/audit_annotation.py \
  dataset/mimic-cxr/annotation_with_progressions_codex.json \
  --output annotation_audit.json
```

The audit reports split sizes, duplicate IDs, missing report/image fields, prompt markers (`User:`, `Assistant:`, `<Image>`, radiologist/edit-program instructions), and correspondence-style drift markers. Add `--fail-on-contamination` for a non-zero exit code when prompt contamination is detected.

## Train

```bash
bash scripts/evoedit_train.sh
```

Example with custom paths:

```bash
ANNOTATION=/data/mimic/annotation_with_progressions_codex.json \
BASE_DIR=/data/mimic-cxr \
VISION_MODEL=/weights/swin-base-patch4-window7-224 \
LLAMA_MODEL=/weights/Llama-2-7b-chat-hf \
DEVICES=2 \
bash scripts/evoedit_train.sh
```

Arguments appended to the script override defaults, while its shared launch values stay synchronized with TIM Stage I.

## Test

```bash
CKPT_FILE=/path/to/checkpoint.ckpt bash scripts/evoedit_test.sh
```

EvoEdit writes generated reports and a finding-wise edit program under `<savedmodel_path>/result/`. Each entry includes the operation, latent anatomy/severity code, calibrated confidence, and preservation gate.

## Checks

```bash
python -m py_compile \
  train.py configs/config.py models/model_evoedit.py models/evoedit/*.py \
  tools/audit_annotation.py
PYTHONPATH=. pytest -q tests/test_evoedit_core.py
bash -n scripts/evoedit_train.sh scripts/evoedit_test.sh
```

Full training requires the restricted MIMIC-CXR data and TIM's pretrained weights.

## Acknowledgement

This repository is built on TIM and inherits TIM's acknowledgement of R2GenGPT. Please cite the original projects when using this code.
