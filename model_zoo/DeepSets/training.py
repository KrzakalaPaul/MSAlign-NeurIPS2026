"""Common trainer for spectrum-to-fingerprint regression baselines."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from .datamodule import FingerprintRegressionDataModule


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _scalar_metrics(metrics):
    return {
        str(key): float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
        for key, value in (metrics[0] if metrics else {}).items()
    }


def train_and_eval_fingerprint_regression(
    args,
    config: dict,
    *,
    model,
    model_name: str,
    spectrum_transform: str,
    spectrum_options: dict,
):
    started_at = time.time()
    seed = int(config.get("seed", 42))
    seed_everything(seed, workers=True)
    training = config["training"]
    data = config.get("data", {})

    datamodule = FingerprintRegressionDataModule(
        labelled_dataset_name=args.labelled_dataset_name,
        candidate_map_name=args.candidate_map_name,
        split_method=args.split_method,
        spectrum_transform=spectrum_transform,
        spectrum_options=spectrum_options,
        batch_size=int(training["batch_size"]),
        batch_size_test=args.batch_size_test,
        n_workers=args.n_workers,
        prefetch_factor=int(data.get("prefetch_factor", 2)),
        data_root=data.get("root", "data"),
    )

    run_name = args.wandb_run_name or model_name
    checkpoint_dir = Path("checkpoints") / "paper_competitors" / run_name.replace("/", "__")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor="loss (val)",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:02d}-{loss (val):.4f}",
        )
    ]
    config_json = json.dumps(config, sort_keys=True, default=str)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    revision = _git_revision()
    logger = False if args.no_logger else WandbLogger(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={**config, **vars(args)},
    )
    trainer = Trainer(
        accelerator=training.get("accelerator", "gpu"),
        max_epochs=int(training["epochs"]),
        gradient_clip_val=float(training.get("gradient_clip_val", 0.0)),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=int(training.get("log_every_n_steps", 10)),
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.fit(model, datamodule=datamodule)
    validation = trainer.validate(model, datamodule=datamodule, ckpt_path="best")
    test = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    return {
        "status": "completed",
        "model": model_name,
        "dataset": args.labelled_dataset_name,
        "candidate_map": args.candidate_map_name,
        "split": args.split_method,
        "model_seed": seed,
        "config_name": args.config,
        "config_hash": config_hash,
        "git_revision": revision,
        "checkpoint_dir": str(checkpoint_dir),
        "num_parameters": model.num_parameters(),
        "elapsed_seconds": time.time() - started_at,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        "evaluation_sizes": {
            "validation": len(datamodule.val_dataset),
            "test": len(datamodule.test_dataset),
        },
        "metrics": {
            "validation": _scalar_metrics(validation),
            "test": _scalar_metrics(test),
        },
    }
