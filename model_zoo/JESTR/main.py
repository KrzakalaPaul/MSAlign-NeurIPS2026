from .datamodule import JESTER_Datamodule
from .model import JESTR
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
import wandb
from lightning.pytorch import seed_everything
from pathlib import Path
import torch
import json


def _existing_pretrain_checkpoint(
    checkpoint_dir: Path, n_epochs_pretrain: int
) -> Path | None:
    """Return a checkpoint only when its matching epoch run completed."""

    marker = checkpoint_dir / "pretrain" / "completed.json"
    try:
        payload = json.loads(marker.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if int(payload.get("n_epochs_pretrain", -1)) != n_epochs_pretrain:
        return None
    checkpoint = Path(payload.get("checkpoint", ""))
    return checkpoint if checkpoint.is_file() else None


def _mark_pretraining_complete(
    checkpoint_dir: Path, checkpoint: Path, n_epochs_pretrain: int
) -> None:
    marker = checkpoint_dir / "pretrain" / "completed.json"
    temporary = marker.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({
        "checkpoint": str(checkpoint.resolve()),
        "n_epochs_pretrain": n_epochs_pretrain,
    }, indent=2) + "\n")
    temporary.replace(marker)


def _result(args, config, validation, test):
    def scalar(metrics):
        return {
            str(key): float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
            for key, value in (metrics[0] if metrics else {}).items()
        }
    return {
        "status": "completed", "model": "JESTR", "dataset": args.labelled_dataset_name,
        "split": args.split_method, "model_seed": int(config.get("seed", 42)),
        "metrics": {"validation": scalar(validation), "test": scalar(test)},
    }

def pretrain_JESTR(args, config):
    checkpoint_dir = Path("checkpoints") / "paper_competitors" / str(
        args.wandb_run_name or "JESTR"
    ).replace("/", "__")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    n_epochs_pretrain = int(config["n_epochs_pretrain"])
    existing_checkpoint = _existing_pretrain_checkpoint(
        checkpoint_dir, n_epochs_pretrain
    )
    if existing_checkpoint is not None:
        print(
            f"Reusing completed JESTR pretraining checkpoint: {existing_checkpoint}",
            flush=True,
        )
        return JESTR.load_from_checkpoint(
            existing_checkpoint,
            config=config,
            weights_only=False,
        )
    model = JESTR(config)
    
    datamodule_pretrain = JESTER_Datamodule(
        labelled_dataset_name=args.labelled_dataset_name,
        candidate_map_name=args.candidate_map_name,
        split_method=args.split_method,
        k_candidates=config['k_candidates'],
        batch_size_test=args.batch_size_test,
        n_workers=args.n_workers,
        bin_width=config['bin_width'],
        max_mz=config['max_mz'],
        batch_size=config['batch_size'],
        mode='pretrain'
    )

    
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir / "pretrain",
            monitor="R@1 - batch (val)",
            mode="max",
            save_top_k=1,
            filename="best-{epoch:02d}-{R@1 - batch (val):.3f}",
            verbose=True,
        ),
    ]

    logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={**config, **vars(args)},
    )

    trainer = Trainer(
        accelerator="gpu", 
        default_root_dir=checkpoint_dir / "pretrain",
        gradient_clip_val=5.0,
        max_epochs=n_epochs_pretrain,
        callbacks=callbacks,
        logger=False if args.no_logger else logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule_pretrain)
    best_model_path = trainer.checkpoint_callback.best_model_path
    _mark_pretraining_complete(
        checkpoint_dir, Path(best_model_path), n_epochs_pretrain
    )
    best_model = JESTR.load_from_checkpoint(best_model_path, config=config, weights_only=False)

    wandb.finish()

    return best_model

def finetune_JESTR(args, config, model):
    checkpoint_dir = Path("checkpoints") / "paper_competitors" / str(
        args.wandb_run_name or "JESTR"
    ).replace("/", "__")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    datamodule_finetune = JESTER_Datamodule(
        labelled_dataset_name=args.labelled_dataset_name,
        candidate_map_name=args.candidate_map_name,
        split_method=args.split_method,
        batch_size_test=args.batch_size_test,
        n_workers=args.n_workers,
        k_candidates=config['k_candidates'],
        bin_width=config['bin_width'],
        max_mz=config['max_mz'],
        batch_size=config['batch_size'],
        mode='finetune'
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir / "finetune",
            monitor="R@1 (val)",
            mode="max",
            save_top_k=1,
            filename="best-finetune-{epoch:02d}-{R@1 (val):.3f}",
            verbose=True,
        ),
    ]

    logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_run_name + "_finetune",
        config={**config, **vars(args)},
    )

    trainer = Trainer(
        accelerator="gpu", 
        default_root_dir=checkpoint_dir / "finetune",
        gradient_clip_val=5.0,
        max_epochs=int(config["n_epochs_finetune"]),
        callbacks=callbacks,
        logger=False if args.no_logger else logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule_finetune)
    validation = trainer.validate(model, datamodule=datamodule_finetune, ckpt_path="best")
    test = trainer.test(model, datamodule=datamodule_finetune, ckpt_path="best")
    
    wandb.finish()
    return _result(args, config, validation, test)
    
def train_and_eval_JESTR(args, config):
    # Candidate batches contain many independently allocated graph tensors.
    # The default descriptor-based IPC can exhaust its descriptor channel with
    # several workers and fail with "received 0 items of ancdata".
    torch.multiprocessing.set_sharing_strategy("file_system")
    seed_everything(int(config.get("seed", 42)), workers=True)
    model = pretrain_JESTR(args, config)
    model.mode = 'finetune' 
    return finetune_JESTR(args, config, model)
