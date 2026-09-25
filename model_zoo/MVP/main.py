"""Training entry point for MVP."""

from pathlib import Path

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import torch

from .datamodule import MVPDataModule
from .model import MVP


def _scalars(output):
    return {str(k): float(v.detach().cpu()) if hasattr(v, "detach") else float(v) for k, v in (output[0] if output else {}).items()}


def train_and_eval_MVP(args, config):
    # Candidate evaluation transfers many independently allocated DGL tensors
    # from workers. Descriptor-backed sharing can exhaust the IPC channel and
    # fail with "received 0 items of ancdata".
    torch.multiprocessing.set_sharing_strategy("file_system")
    seed_everything(int(config.get("seed", 0)), workers=True)
    checkpoint_dir = Path("checkpoints/paper_competitors") / str(args.wandb_run_name or "MVP").replace("/", "__")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    evaluation_only = bool(config.get("evaluation_only", False))
    existing_checkpoints = list(checkpoint_dir.glob("best-*.ckpt"))
    if evaluation_only and not existing_checkpoints:
        raise FileNotFoundError(
            f"Evaluation-only MVP run has no checkpoint in {checkpoint_dir}."
        )
    data = MVPDataModule(
        labelled_dataset_name=args.labelled_dataset_name,
        candidate_map_name=args.candidate_map_name,
        split_method=args.split_method,
        config=config,
        n_workers=args.n_workers,
        batch_size_test=args.batch_size_test,
        evaluation_only=evaluation_only,
    )
    if evaluation_only:
        checkpoint_path = max(
            existing_checkpoints, key=lambda path: path.stat().st_mtime_ns
        )
        print(f"Reusing completed MVP checkpoint: {checkpoint_path}", flush=True)
        model = MVP.load_from_checkpoint(
            checkpoint_path, config=config, weights_only=False
        )
    else:
        model = MVP(config)
    checkpoint = ModelCheckpoint(dirpath=checkpoint_dir, monitor="loss (train)", mode="min", save_top_k=1, filename="best-{epoch:04d}")
    logger = WandbLogger(project=args.wandb_project, name=args.wandb_run_name, config={**config, **vars(args)})
    trainer = Trainer(
        accelerator=config["training"]["accelerator"],
        max_epochs=config["training"]["max_epochs"],
        callbacks=[checkpoint], logger=False if args.no_logger else logger,
        log_every_n_steps=config["training"]["log_every_n_steps"],
        num_sanity_val_steps=0,
    )
    if evaluation_only:
        validation = []
    else:
        trainer.fit(model, datamodule=data)
        checkpoint_path = Path(checkpoint.best_model_path)
        model = MVP.load_from_checkpoint(
            checkpoint_path, config=config, weights_only=False
        )
        validation = trainer.validate(model, datamodule=data)
    test = trainer.test(model, datamodule=data)
    return {
        "status": "completed", "model": "MVP", "dataset": args.labelled_dataset_name,
        "split": args.split_method, "model_seed": int(config.get("seed", 0)),
        "spectrum_mode": config["spectrum_mode"],
        "oracle_assisted": config["spectrum_mode"] == "formula",
        "ranking_view": "mol_spec",
        "metrics": {"validation": _scalars(validation), "test": _scalars(test)},
    }
