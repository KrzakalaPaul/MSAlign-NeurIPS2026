from .datamodule import EmbCos_Datamodule
from .model import EmbCos
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch import seed_everything
from pathlib import Path


def _result(args, config, validation, test):
    def scalar(metrics):
        return {
            str(key): float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
            for key, value in (metrics[0] if metrics else {}).items()
        }
    return {
        "status": "completed", "model": "EmbCos", "dataset": args.labelled_dataset_name,
        "split": args.split_method, "model_seed": int(config.get("seed", 42)),
        "metrics": {"validation": scalar(validation), "test": scalar(test)},
    }

def train_and_eval_EmbCos(args, config):

    seed_everything(int(config.get("seed", 42)), workers=True)
    checkpoint_dir = Path("checkpoints") / "paper_competitors" / str(
        args.wandb_run_name or "EmbCos"
    ).replace("/", "__")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    datamodule = EmbCos_Datamodule(
        labelled_dataset_name=args.labelled_dataset_name,
        candidate_map_name=args.candidate_map_name,
        split_method=args.split_method,
        batch_size_test=args.batch_size_test,
        n_workers=args.n_workers,
        k_candidates=config['k_candidates'],
        fingerprint_size=config['fingerprint_size'],
        bin_width=config['bin_width'],
        max_mz=config['max_mz'],
        batch_size=config['batch_size'],
    )

    model = EmbCos(config)

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor="R@1 (val)",
            mode="max",
            save_top_k=1,
            filename="best-{epoch:02d}-{R@1 (val):.3f}",
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
        default_root_dir=checkpoint_dir,
        gradient_clip_val=5.0,
        max_steps=config['n_max_steps'],
        callbacks=callbacks,
        logger=False if args.no_logger else logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)
    validation = trainer.validate(model, datamodule=datamodule, ckpt_path="best")
    test = trainer.test(model, datamodule=datamodule, ckpt_path="best")
    return _result(args, config, validation, test)
