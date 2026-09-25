"""Small CSV cache for sliced Wasserstein results."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


OUTPUT_PATH = Path("sliced_wasserstein/results.csv")
FIELDS = ("dataset", "split", "ms_representation", "mol_representation", "SWD")
LEGACY_FIELDS = ("dataset", "split", "SWD")


def read_results() -> list[dict[str, str]]:
    if not OUTPUT_PATH.exists():
        return []
    with OUTPUT_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if tuple(reader.fieldnames or ()) == FIELDS:
        return rows
    if tuple(reader.fieldnames or ()) == LEGACY_FIELDS:
        return [
            {
                "dataset": row["dataset"],
                "split": row["split"],
                "ms_representation": "unknown",
                "mol_representation": "unknown",
                "SWD": row["SWD"],
            }
            for row in rows
        ]
    raise ValueError(f"Unexpected columns in {OUTPUT_PATH}: {reader.fieldnames}")


def get_result(
    dataset: str,
    split: str,
    ms_representation: str,
    mol_representation: str,
) -> float | None:
    for row in read_results():
        if (
            row["dataset"] == dataset
            and row["split"] == split
            and row["ms_representation"] == ms_representation
            and row["mol_representation"] == mol_representation
        ):
            return float(row["SWD"])
    return None


def write_result(
    dataset: str,
    split: str,
    ms_representation: str,
    mol_representation: str,
    distance: float,
) -> None:
    """Append one result, replacing an existing row for the same encoders."""

    rows = read_results()
    new_row = {
        "dataset": dataset,
        "split": split,
        "ms_representation": ms_representation,
        "mol_representation": mol_representation,
        "SWD": str(distance),
    }
    for index, row in enumerate(rows):
        if all(row[field] == new_row[field] for field in FIELDS[:-1]):
            rows[index] = new_row
            break
    else:
        rows.append(new_row)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(OUTPUT_PATH)


def normalized_swd(distance: float, random_distances: list[float]) -> float:
    """Normalize one SWD by the mean SWD of the five random splits."""

    if not random_distances:
        raise ValueError("At least one random-split distance is required.")
    denominator = float(np.mean(random_distances))
    if denominator <= 0:
        raise ValueError("The mean random-split SWD must be positive.")
    return distance / denominator
