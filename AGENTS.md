# Repository Guidelines

## Project Structure & Module Organization

`train.py` is the unified PyTorch Lightning entry point. EvoEdit-specific code lives in `models/evoedit/`: `program.py` builds edit programs, `targets.py` derives operation targets, `losses.py` and `copy.py` implement objectives, and the `model_*` modules separate core, training, and generation behavior. `models/model_evoedit.py` assembles those components. CLI defaults are in `configs/config.py`; runnable wrappers are under `scripts/`. Unit tests live in `tests/`. The pinned `TIM/` submodule supplies upstream data modules, utilities, and stage 1/2 implementations through tracked links.

Restricted MIMIC-CXR data belongs in `dataset/mimic-cxr/`, pretrained models in `pretrain_weights/`, and run artifacts in `save/`; all are intentionally ignored by Git.

## Build, Test, and Development Commands

Use Python 3.9 and initialize the upstream dependency before installing:

```bash
git submodule update --init --recursive
conda create -n evoedit python=3.9 && conda activate evoedit
pip install -r requirements.txt
```

Run focused checks before submitting changes:

```bash
python -m py_compile train.py configs/config.py models/model_evoedit.py models/evoedit/*.py
pytest -q tests/test_evoedit_core.py
bash -n scripts/evoedit_train.sh scripts/evoedit_test.sh
```

Start training with `bash scripts/evoedit_train.sh`. Evaluate with `CKPT_FILE=/path/model.ckpt bash scripts/evoedit_test.sh`. Override local paths through uppercase environment variables such as `ANNOTATION`, `BASE_DIR`, and `VISION_MODEL`.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, type hints, short docstrings, and `from __future__ import annotations`. Use `snake_case` for functions and variables, `PascalCase` for classes, and uppercase names for constants and shell configuration variables. Keep model responsibilities in the existing focused modules. No formatter is configured, so avoid unrelated formatting churn. Shell scripts must retain `set -euo pipefail` and quote path variables.

## Testing Guidelines

Pytest is the unit-test framework. Name files `test_*.py` and functions `test_*`. Add deterministic, CPU-sized tests for target logic, tensor shapes, algebraic invariants, and finite losses; seed Torch where randomness is used. No coverage threshold is configured. Full training is not a CI requirement because it needs GPUs, restricted data, and pretrained weights.

## Commit & Pull Request Guidelines

The current history uses a Conventional Commit-style subject (`feat: ...`). Continue with concise imperative prefixes such as `feat:`, `fix:`, `test:`, or `docs:`. Pull requests should explain the behavioral change, identify affected model/config paths, link any issue, and list exact validation commands and results. Include representative generation output when report behavior changes. Call out TIM submodule-pointer changes explicitly; never commit patient data, model weights, checkpoints, or logs.
