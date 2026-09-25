"""Calibrate MSAlign score fusion on validation and evaluate it on test."""

from __future__ import annotations

import argparse
from pathlib import Path

from models.MSAlign_fusion import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("models/MSAlign_fusion/configs/default.yaml"),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output, rows = run_experiment(args.config, verbose=args.verbose)
    print(f"Wrote {output}")
    for row in rows:
        metrics = ", ".join(
            f"{metric}={row[metric]:.4f}" for metric in ("R@1", "R@5", "R@20")
        )
        print(
            f"{row['labelled_dataset_name']}/{row['split_method']} "
            f"{row['name']}: {metrics}"
        )


if __name__ == "__main__":
    main()
