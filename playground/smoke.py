"""Run one raw-input score through an extracted playground bundle."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from .inference import MSAlign


SPECTRUM = {
    "peaks": [
        [43.0184, 0.18],
        [77.0390, 0.12],
        [105.0335, 0.46],
        [121.0284, 1.00],
        [138.0550, 0.31],
    ],
    "precursor_mz": 181.0495,
    "adduct": "[M+H]+",
    "collision_energy": 30.0,
}
CANDIDATES = ("CC(=O)Oc1ccccc1C(=O)O", "CCO")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    model = MSAlign(args.bundle)
    single = model(SPECTRUM, CANDIDATES[0])
    multiple = model(SPECTRUM, list(CANDIDATES))
    if not isinstance(single, float) or not math.isfinite(single):
        raise RuntimeError("Single-candidate inference did not return a finite float")
    if len(multiple) != 2 or not all(math.isfinite(value) for value in multiple):
        raise RuntimeError("Multi-candidate inference returned invalid scores")
    if not math.isclose(single, multiple[0], rel_tol=1e-5, abs_tol=1e-5):
        raise RuntimeError("Candidate-list scoring changed the first candidate score")

    without_metadata = {
        "peaks": SPECTRUM["peaks"],
        "precursor_mz": SPECTRUM["precursor_mz"],
    }
    if not math.isfinite(model(without_metadata, CANDIDATES[0])):
        raise RuntimeError("Inference with optional metadata omitted was not finite")
    try:
        model(SPECTRUM, "not-a-smiles")
    except ValueError:
        pass
    else:
        raise RuntimeError("Invalid SMILES was not rejected")
    print(f"{args.bundle.name}: single={single:.6f}, candidates={multiple}")


if __name__ == "__main__":
    main()
