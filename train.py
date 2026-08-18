"""Unified TIM/EvoEdit training entry point."""

from __future__ import annotations

import importlib
import os
from pprint import pprint

import lightning.pytorch as pl
import torch
from lightning.pytorch.strategies import DeepSpeedStrategy

from configs.config import parser
from dataset.longitudinal_data_module import DataModule
from lightning_tools.callbacks import add_callbacks


torch.set_float32_matmul_precision("medium")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _build_strategy(args):
    if args.strategy == "deepspeed":
        return DeepSpeedStrategy(config=args.deepspeed_config)
    if args.strategy == "auto" or args.devices == 1:
        return "auto"
    return args.strategy


def train(args) -> None:
    callbacks = add_callbacks(args)
    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=_build_strategy(args),
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval=args.val_check_interval,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        max_epochs=args.max_epochs,
        num_sanity_val_steps=args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks["callbacks"],
        logger=callbacks["loggers"],
    )
    data_module = DataModule(args)
    model_module = importlib.import_module(f"models.model_{args.stage}")
    model_class = model_module.LongitudinalR2GenGPT
    model = (
        model_class.load_from_checkpoint(args.ckpt_file, strict=False, args=args)
        if args.ckpt_file
        else model_class(args)
    )
    if args.test:
        trainer.test(model, datamodule=data_module)
    elif args.validate:
        trainer.validate(model, datamodule=data_module)
    else:
        trainer.fit(model, datamodule=data_module)


def main() -> None:
    args = parser.parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    pl.seed_everything(42, workers=True)
    train(args)


if __name__ == "__main__":
    main()
