"""Compare standalone scorer channels with training-time representations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from playground.inference import (
    MSAlign,
    _adduct_offset,
    _binned_spectrum,
    _canonical_smiles,
    _combine_channels,
)


def _column(path: Path, name: str) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row[name] for row in csv.DictReader(handle)]


def _first_test_row(root: Path, split: str) -> int:
    folds = _column(root / "splits" / f"{split}.csv", "fold")
    return folds.index("test")


def _cached_molecules(
    candidate_root: Path, representation: str, target_index: int, size: int
) -> torch.Tensor:
    key = (
        "candidate_fingerprints"
        if representation == "morgan_2_4096"
        else "candidates_embeddings"
    )
    with h5py.File(candidate_root / f"{representation}.h5") as handle:
        values = np.asarray(handle[key][target_index, :size])
    if representation == "morgan_2_4096":
        values = np.unpackbits(values, axis=-1, count=4096, bitorder="little")
    return torch.from_numpy(np.array(values, copy=True)).float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--split", choices=("formula_seed0", "mces_1"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/massspecgym"))
    args = parser.parse_args()

    model = MSAlign(args.bundle)
    row = _first_test_row(args.data_root, args.split)
    metadata = list(csv.DictReader((args.data_root / "metadata.csv").open()))[row]
    target_index = int(metadata["unique_smiles_idx"])
    target = _column(args.data_root / "unique_smiles.csv", "smiles")[target_index]
    with (args.data_root / "candidates/official_candidates_by_mass/map.json").open() as handle:
        candidates = json.load(handle)[target][:4]

    peaks = np.load(args.data_root / "spectra.npy", allow_pickle=True)[row]
    raw_energy = metadata.get("collision_energy")
    try:
        energy = float(raw_energy)
    except (TypeError, ValueError):
        energy = float("nan")
    spectrum = {
        "peaks": peaks,
        "precursor_mz": float(metadata["precursor_mz"]),
        "adduct": metadata.get("adduct") or None,
        "collision_energy": energy if np.isfinite(energy) else None,
    }
    online = model._score_channels(spectrum, candidates)
    mask = torch.ones((1, len(candidates)), dtype=torch.bool, device=model.device)
    candidate_root = args.data_root / "candidates/official_candidates_by_mass"

    for scorer in model.alignment_scorers:
        config = scorer["config"]
        ms_name = config["representations"]["spectrum"]
        mol_name = config["representations"]["molecule"]
        if ms_name == "dreams":
            spectrum_values = np.load(
                args.data_root / "spectra_embeddings/dreams.npy", mmap_mode="r"
            )[row]
        else:
            spectrum_values = _binned_spectrum(peaks, **config["spectrum_bins"])
        molecule_values = _cached_molecules(
            candidate_root, mol_name, target_index, len(candidates)
        )
        metadata_values = model._metadata(
            scorer["dimensions"],
            spectrum["adduct"],
            (
                float("nan")
                if spectrum["collision_energy"] is None
                else spectrum["collision_energy"]
            ),
            model.device,
        )
        cached = scorer["model"](
            torch.as_tensor(spectrum_values, device=model.device)[None],
            molecule_values.to(model.device)[None],
            metadata_values,
            mask,
        )[0]
        torch.testing.assert_close(
            online[scorer["name"]], cached, rtol=2e-4, atol=2e-5
        )

    estimated = spectrum["precursor_mz"] - _adduct_offset(spectrum["adduct"])
    if model.mass_scorers and np.isfinite(estimated):
        from rdkit import Chem
        from rdkit.Chem.Descriptors import ExactMolWt

        masses = torch.tensor(
            [ExactMolWt(Chem.MolFromSmiles(value)) for value in candidates],
            device=model.device,
        )
        ppm = 1e6 * (masses - estimated).abs() / masses
        for scorer in model.mass_scorers:
            expected = torch.exp(-0.5 * (ppm / float(scorer["scale_ppm"])).square())
            torch.testing.assert_close(online[scorer["name"]], expected)

    public = torch.tensor(model(spectrum, candidates), device=model.device)
    torch.testing.assert_close(
        public, _combine_channels(model.manifest, online), rtol=2e-4, atol=2e-5
    )
    print(
        f"Verified {args.bundle.name} against cached representations on "
        f"{args.split} test row {row}."
    )


if __name__ == "__main__":
    main()
