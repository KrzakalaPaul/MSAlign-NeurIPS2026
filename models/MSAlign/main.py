"""Train MSAlign and save the checkpoint selected on validation R@1."""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from .datamodule import MSAlignDataModule
from .model import MSAlign


def build_datamodule(args, config: dict) -> MSAlignDataModule:
    model = config["model"]
    training = config["training"]
    return MSAlignDataModule(
        labelled_dataset_name=args.labelled_dataset_name,
        candidate_map_name=args.candidate_map_name,
        split_method=args.split_method,
        k_candidates=training["k_candidates"],
        ms_representation=config["representations"]["spectrum"],
        mol_representation=config["representations"]["molecule"],
        use_energy=model.get("use_collision_energy", True),
        use_adduct=model.get("use_adduct", True),
        spectra_bins=config.get("spectrum_bins", {"max_mz": 1005.0, "bin_width": 0.1}),
        batch_size=training["batch_size"],
        batch_size_test=args.batch_size_test,
        n_workers=args.n_workers,
        data_root=config.get("data_root", "data"),
    )


def train_MSAlign(args, config: dict) -> Path:
    """Fit one model and return the checkpoint selected on validation R@1."""

    seed_everything(int(config.get("seed", 42)), workers=True)
    data = build_datamodule(args, config)
    model = MSAlign(config, data.input_dimensions)

    run_name = args.wandb_run_name or (
        f"{args.labelled_dataset_name}-{args.split_method}-"
        f"{config['representations']['spectrum']}-{config['representations']['molecule']}"
    )
    requested_checkpoint = getattr(args, "output_checkpoint", None)
    checkpoint_path = Path(requested_checkpoint) if requested_checkpoint else None
    checkpoint_dir = checkpoint_path.parent if checkpoint_path else Path("checkpoints") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / "config.yaml").open("w") as handle:
        import yaml
        yaml.safe_dump(config, handle, sort_keys=False)

    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=checkpoint_path.stem if checkpoint_path else "best",
        monitor="R@1/val",
        mode="max",
        save_top_k=1,
    )
    logger = False if args.no_logger else WandbLogger(
        project=args.wandb_project, name=run_name, config=config
    )
    training = config["training"]
    trainer = Trainer(
        accelerator=training.get("accelerator", "gpu"),
        max_steps=training["n_max_steps"],
        gradient_clip_val=training.get("gradient_clip_val", 5.0),
        callbacks=[checkpoint],
        logger=logger,
        log_every_n_steps=training.get("log_every_n_steps", 10),
    )
    trainer.fit(model, datamodule=data)
    best = Path(checkpoint.best_model_path)
    print(f"Best checkpoint: {best}")
    return best
