"""Single-config evaluation interface for MSAlign_fusion."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

import torch
import yaml

from .model import MSAlignFusion, retrieval_metrics
from .formula_filter import candidate_formula_mask


CONFIG_KEYS = {
    "fit",
    "scorers",
    "output_csv",
}
OPTIONAL_CONFIG_KEYS = {"cache_dir"}
OPTIONAL_CONFIG_KEYS.add("evaluation_targets")
OPTIONAL_CONFIG_KEYS.add("formula_filter")

EVALUATION_TARGETS = (
    {
        "labelled_dataset_name": "massspecgym",
        "candidate_map_name": "official_candidates_by_mass",
        "split_method": "formula_seed1",
    },
    {
        "labelled_dataset_name": "massspecgym",
        "candidate_map_name": "official_candidates_by_mass",
        "split_method": "formula_seed2",
    },
    {
        "labelled_dataset_name": "massspecgym",
        "candidate_map_name": "official_candidates_by_mass",
        "split_method": "formula_seed3",
    },
    {
        "labelled_dataset_name": "spectraverse",
        "candidate_map_name": "256_candidates_by_mass",
        "split_method": "formula_seed1",
    },
    {
        "labelled_dataset_name": "spectraverse",
        "candidate_map_name": "256_candidates_by_mass",
        "split_method": "formula_seed2",
    },
    {
        "labelled_dataset_name": "spectraverse",
        "candidate_map_name": "256_candidates_by_mass",
        "split_method": "formula_seed3",
    },
)


def load_experiment_config(path: str | Path) -> dict:
    """Load and validate the top-level experiment configuration."""

    path = Path(path)
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The experiment config must be a YAML mapping.")
    missing = CONFIG_KEYS - set(config)
    unknown = set(config) - CONFIG_KEYS - OPTIONAL_CONFIG_KEYS
    if missing:
        raise ValueError(f"Experiment config is missing: {sorted(missing)}.")
    if unknown:
        raise ValueError(f"Unknown experiment config fields: {sorted(unknown)}.")
    if not isinstance(config["fit"], dict):
        raise ValueError("fit must be a mapping.")
    if not isinstance(config["scorers"], list) or not config["scorers"]:
        raise ValueError("scorers must be a non-empty list.")
    if not isinstance(config["output_csv"], str) or not config["output_csv"]:
        raise ValueError("output_csv must be a non-empty path string.")
    cache_dir = config.get("cache_dir", "cache/MSAlign_fusion")
    if cache_dir is not None and (not isinstance(cache_dir, str) or not cache_dir):
        raise ValueError("cache_dir must be a non-empty path string or null.")
    targets = config.get("evaluation_targets")
    if targets is not None:
        if not isinstance(targets, list) or not targets:
            raise ValueError("evaluation_targets must be a non-empty list.")
        required = {
            "labelled_dataset_name",
            "candidate_map_name",
            "split_method",
        }
        for index, target in enumerate(targets):
            if not isinstance(target, dict) or set(target) != required:
                raise ValueError(
                    f"evaluation_targets[{index}] must contain exactly "
                    f"{sorted(required)}."
                )
            if any(not isinstance(target[key], str) or not target[key] for key in required):
                raise ValueError(
                    f"evaluation_targets[{index}] values must be non-empty strings."
                )
    return config


def _metric_values(metrics: Mapping[str, torch.Tensor]) -> dict[str, float | None]:
    return {
        name: float(metrics[name].detach().cpu()) if name in metrics else None
        for name in ("R@1", "R@5", "R@20")
    }


def evaluate_config(
    config: Mapping[str, object], *, verbose: bool = False
) -> list[dict[str, object]]:
    """Fit and evaluate every fixed dataset/split target."""

    model = MSAlignFusion(
        config["scorers"],
        cache_dir=config.get("cache_dir", "cache/MSAlign_fusion"),
    )
    rows = []
    targets = config.get("evaluation_targets", EVALUATION_TARGETS)
    for index, datasets_params in enumerate(targets, start=1):
        if verbose:
            print(
                f"[{index}/{len(targets)}] "
                f"{datasets_params['labelled_dataset_name']} / "
                f"{datasets_params['split_method']}",
                flush=True,
            )
        rows.extend(
            _evaluate_target(
                model, datasets_params, config["fit"], verbose=verbose,
                formula_filter=bool(config.get("formula_filter", False)),
            )
        )
    return rows


def _evaluate_target(
    model: MSAlignFusion,
    datasets_params: Mapping[str, str],
    fit_params: Mapping[str, object],
    *,
    verbose: bool = False,
    formula_filter: bool = False,
    include_scorers: bool = True,
) -> list[dict[str, object]]:
    """Fit on validation and evaluate test, optionally with an oracle filter."""

    validation_scores, validation_mask = model.get_scores(
        datasets_params, fold="val", verbose=verbose
    )
    calibration_mask = validation_mask
    if formula_filter:
        if verbose:
            print(
                "  applying oracle molecular-formula filter to validation",
                flush=True,
            )
        calibration_mask = candidate_formula_mask(
            datasets_params, "val", validation_mask
        )
    if verbose:
        print("  fitting calibration weights", flush=True)
    alpha = model.fit(validation_scores, calibration_mask, fit_params)

    test_scores, test_mask = model.get_scores(
        datasets_params, fold="test", verbose=verbose
    )
    evaluation_mask = test_mask
    if formula_filter:
        if verbose:
            print("  applying oracle molecular-formula filter to test", flush=True)
        evaluation_mask = candidate_formula_mask(datasets_params, "test", test_mask)
    combined = model.eval(test_scores, evaluation_mask)
    common = {
        "labelled_dataset_name": datasets_params["labelled_dataset_name"],
        "candidate_map_name": datasets_params["candidate_map_name"],
        "split_method": datasets_params["split_method"],
        "n_test": int(test_scores.shape[0]),
        "selected_temperature": model.selected_temperature,
        "evaluation_filter": "molecular_formula" if formula_filter else "none",
    }
    alpha_by_name = {
        name: float(value)
        for name, value in zip(model.scorer_names, alpha.detach().cpu())
    }
    rows = [
        {
            **common,
            "name": "MSAlign_fusion",
            "kind": "model",
            "scorer_type": "combined",
            "alpha": json.dumps(alpha_by_name, sort_keys=True),
            **_metric_values(combined["metrics"]),
        }
    ]
    if not include_scorers:
        return rows
    for index, scorer in enumerate(model.scorers):
        metrics = retrieval_metrics(test_scores[:, :, index], evaluation_mask)
        rows.append(
            {
                **common,
                "name": scorer.name,
                "kind": "scorer",
                "scorer_type": scorer.scorer_type,
                "alpha": float(alpha[index].detach().cpu()),
                **_metric_values(metrics),
            }
        )
    return rows


def write_results(rows: list[dict[str, object]], path: str | Path) -> Path:
    """Atomically write the compact test-results table."""

    if not rows:
        raise ValueError("Cannot write an empty result table.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = [
        "labelled_dataset_name",
        "candidate_map_name",
        "split_method",
        "n_test",
        "selected_temperature",
        "evaluation_filter",
        "name",
        "kind",
        "scorer_type",
        "alpha",
        "R@1",
        "R@5",
        "R@20",
    ]
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def run_experiment(
    config_path: str | Path, *, verbose: bool = False
) -> tuple[Path, list[dict[str, object]]]:
    """Run one YAML-defined MSAlign fusion calibration/evaluation."""

    config = load_experiment_config(config_path)
    rows = evaluate_config(config, verbose=verbose)
    output = write_results(rows, config["output_csv"])
    return output, rows
