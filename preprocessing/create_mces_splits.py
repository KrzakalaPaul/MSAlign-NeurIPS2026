#!/usr/bin/env python3
"""Create two explicit MCES splits from MassSpecGym's provided split."""

from __future__ import annotations

import argparse
import fcntl
from pathlib import Path

import pandas as pd


def _write_split(path: Path, folds: pd.Series, *, overwrite: bool) -> None:
    output = pd.DataFrame({"fold": folds})
    if path.is_file() and not overwrite:
        existing = pd.read_csv(path)
        if existing.equals(output):
            print(f"Skipping unchanged split: {path}")
            return
        raise FileExistsError(
            f"Split already exists with different contents: {path}. "
            "Pass --overwrite to replace it."
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(path)
    print(f"Wrote {path}: {output['fold'].value_counts().to_dict()}")


def create_mces_splits(
    data_root: str | Path = "data",
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Create `mces_1` and its validation/test-swapped counterpart `mces_2`."""

    split_dir = Path(data_root) / "massspecgym" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    first_path = split_dir / "mces_1.csv"
    second_path = split_dir / "mces_2.csv"
    with (split_dir / ".mces.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        source_path = split_dir / "as_provided.csv"
        source = pd.read_csv(source_path)
        if list(source.columns) != ["fold"]:
            raise ValueError(f"Expected one 'fold' column in {source_path}.")
        if source["fold"].isna().any():
            raise ValueError(f"Missing fold values in {source_path}.")
        observed = set(source["fold"].unique())
        expected = {"train", "val", "test"}
        if observed != expected:
            raise ValueError(
                f"Expected folds {sorted(expected)}, found {sorted(observed)}."
            )

        mces_1 = source["fold"].copy()
        mces_2 = source["fold"].replace({"val": "test", "test": "val"})
        _write_split(first_path, mces_1, overwrite=overwrite)
        _write_split(second_path, mces_2, overwrite=overwrite)
    return first_path, second_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    create_mces_splits(args.data_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
