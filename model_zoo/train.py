"""Train one of the five independently reproduced baselines.

The model zoo is intentionally separate from MSAlign: it shares the processed
datasets and retrieval metrics, but none of its architectures are imported by
the main method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


MODELS = {
    "deepsets": ("DeepSets", "train_and_eval_DeepSet"),
    "embcos": ("EmbCos", "train_and_eval_EmbCos"),
    "jestr": ("JESTR", "train_and_eval_JESTR"),
    "flare": ("FLARE", "train_and_eval_FLARE"),
    "mvp": ("MVP", "train_and_eval_MVP"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(MODELS))
    parser.add_argument("--dataset", dest="labelled_dataset_name", required=True)
    parser.add_argument("--candidate-map", dest="candidate_map_name", required=True)
    parser.add_argument("--split", dest="split_method", required=True)
    parser.add_argument("--config", default="default")
    parser.add_argument("--result-path")
    parser.add_argument("--workers", dest="n_workers", type=int, default=8)
    parser.add_argument("--test-batch-size", dest="batch_size_test", type=int, default=16)
    parser.add_argument("--wandb-project", default="MSAlign-model-zoo")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--no-logger", action="store_true")
    parser.add_argument("--resume-checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.wandb_run_name is None:
        args.wandb_run_name = f"{args.model}/{args.labelled_dataset_name}/{args.split_method}"
    package, function_name = MODELS[args.model]
    config_path = Path("model_zoo") / package / "configs" / f"{args.config}.yaml"
    with config_path.open() as handle:
        config = yaml.safe_load(handle)

    try:
        module = __import__(f"model_zoo.{package}.main", fromlist=[function_name])
    except (ImportError, OSError) as error:
        if args.model in {"jestr", "flare", "mvp"}:
            raise RuntimeError(
                f"{package} requires the CUDA graph-baseline dependencies. "
                "Install them with `uv sync --extra graph-baselines` and use "
                "a CUDA 12.1 runtime compatible with the released DGL wheel."
            ) from error
        raise
    result = getattr(module, function_name)(args, config)
    if args.result_path and result is not None:
        output = Path(args.result_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
