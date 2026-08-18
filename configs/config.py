"""Command-line configuration for TIM and EvoEdit."""

from __future__ import annotations

import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


parser = argparse.ArgumentParser(description="EvoEdit on the TIM longitudinal RRG codebase")

# Dataset: identical inputs and splits to TIM.
parser.add_argument("--test", action="store_true", help="only run the test set")
parser.add_argument("--validate", action="store_true", help="only run the validation set")
parser.add_argument("--dataset", type=str, default="longitudinal-mimic")
parser.add_argument("--annotation", type=str, default="./dataset/mimic-cxr/annotation.json")
parser.add_argument("--base_dir", type=str, default="./dataset/mimic-cxr/")
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--val_batch_size", default=4, type=int)
parser.add_argument("--test_batch_size", default=4, type=int)
parser.add_argument("--prefetch_factor", default=4, type=int)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--longitudinal", default=True, type=str2bool, help="TIM compatibility flag")

# Backbone and parameter-efficient tuning.
parser.add_argument("--vision_model", default="./pretrain_weights/swin-base-patch4-window7-224/", type=str)
parser.add_argument("--llama_model", default="./pretrain_weights/Llama-2-7b-chat-hf/", type=str)
parser.add_argument("--freeze_vm", default=True, type=str2bool)
parser.add_argument("--llm_use_lora", default=False, type=str2bool)
parser.add_argument("--llm_r", default=16, type=int)
parser.add_argument("--llm_alpha", default=16, type=int)
parser.add_argument("--vis_use_lora", default=False, type=str2bool)
parser.add_argument("--vis_r", default=16, type=int)
parser.add_argument("--vis_alpha", default=16, type=int)
parser.add_argument("--lora_dropout", default=0.1, type=float)
parser.add_argument("--global_only", default=False, type=str2bool)
parser.add_argument("--low_resource", default=False, type=str2bool)
parser.add_argument("--end_sym", default="</s>", type=str)
parser.add_argument("--stage", default="evoedit", choices=["evoedit", "stage1", "stage2"])
parser.add_argument("--max_iteration", default=3, type=int)

# EvoEdit architecture.
parser.add_argument("--evoedit_heads", default=8, type=int)
parser.add_argument("--evoedit_dropout", default=0.1, type=float)
parser.add_argument("--operation_temperature", default=1.0, type=float)
parser.add_argument("--num_anatomy_codes", default=8, type=int)
parser.add_argument("--num_severity_codes", default=3, type=int)
parser.add_argument("--prompt_max_length", default=320, type=int)

# EvoEdit objective weights.
parser.add_argument("--prior_report_weight", default=0.5, type=float)
parser.add_argument("--operation_loss_weight", default=0.5, type=float)
parser.add_argument("--inverse_loss_weight", default=0.1, type=float)
parser.add_argument("--cycle_loss_weight", default=0.05, type=float)
parser.add_argument("--verifier_loss_weight", default=0.2, type=float)
parser.add_argument("--confidence_loss_weight", default=0.05, type=float)
parser.add_argument("--intervention_loss_weight", default=0.1, type=float)
parser.add_argument("--sparsity_loss_weight", default=0.01, type=float)
parser.add_argument("--factor_balance_weight", default=0.01, type=float)
parser.add_argument("--copy_loss_weight", default=0.1, type=float)
parser.add_argument("--pathology_loss_weight", default=0.0, type=float)

# Checkpoints and model selection.
parser.add_argument("--savedmodel_path", type=str, default="./save/longitudinal-mimic/evoedit")
parser.add_argument("--ckpt_file", type=str, default=None)
parser.add_argument("--delta_file", type=str, default=None)
parser.add_argument("--weights", nargs="+", type=float, default=[0.5, 0.5])
parser.add_argument("--scorer_types", nargs="+", default=["Bleu_4", "CIDEr"])

# Optimization.
parser.add_argument("--learning_rate", default=1e-4, type=float)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--gradient_clip_val", default=1.0, type=float)

# Decoding.
parser.add_argument("--beam_size", type=int, default=3)
parser.add_argument("--do_sample", type=str2bool, default=False)
parser.add_argument("--no_repeat_ngram_size", type=int, default=2)
parser.add_argument("--num_beam_groups", type=int, default=1)
parser.add_argument("--min_new_tokens", type=int, default=40)
parser.add_argument("--max_new_tokens", type=int, default=150)
parser.add_argument("--max_length", type=int, default=150)
parser.add_argument("--repetition_penalty", type=float, default=2.0)
parser.add_argument("--length_penalty", type=float, default=2.0)
parser.add_argument("--diversity_penalty", type=float, default=0.0)
parser.add_argument("--temperature", type=float, default=0.0)

# PyTorch Lightning.
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--accelerator", type=str, default="gpu", choices=["cpu", "gpu", "tpu", "ipu", "hpu", "mps"])
parser.add_argument("--strategy", type=str, default="deepspeed", help="deepspeed, ddp, or auto")
parser.add_argument("--precision", type=str, default="bf16-mixed")
parser.add_argument("--limit_val_batches", type=float, default=1.0)
parser.add_argument("--limit_test_batches", type=float, default=1.0)
parser.add_argument("--limit_train_batches", type=float, default=1.0)
parser.add_argument("--max_epochs", type=int, default=3)
parser.add_argument("--every_n_train_steps", type=int, default=0)
parser.add_argument("--val_check_interval", type=float, default=1.0)
parser.add_argument("--accumulate_grad_batches", type=int, default=1)
parser.add_argument("--num_sanity_val_steps", type=int, default=2)
parser.add_argument("--deepspeed_config", type=str, default="configs/deepspeed.json")
