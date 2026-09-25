# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "dreams @ git+https://github.com/pluskal-lab/DreaMS.git@dbec3a0b514a99e5056cfccde4559fda8cfe8129",
# ]
# ///
"""Generate DreaMS embeddings in an isolated dependency environment.

Run this file through ``precompute_representations.py``. Its inline dependency
metadata is intentionally separate from the main project: the official DreaMS
release pins an older scientific-Python stack and the legacy ``rdkit-pypi``
distribution, which must not modify MSAlign's environment.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from dreams.api import PreTrainedModel
from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
N_HIGHEST_PEAKS = 100


class SpectraDataset(Dataset):
    def __init__(self, spectra, metadata):
        self.spectra = spectra
        self.metadata = metadata
        self.preprocessor = SpectrumPreprocessor(
            DataFormatA(), n_highest_peaks=N_HIGHEST_PEAKS
        )

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, index):
        precursor_mz = self.metadata.iloc[index]["precursor_mz"]
        processed = self.preprocessor(self.spectra[index], precursor_mz)
        return torch.tensor(processed, dtype=torch.float32)


def encode(dataset_name, batch_size, workers, overwrite):
    output = ROOT / "data" / dataset_name / "spectra_embeddings" / "dreams.npy"
    if output.exists() and not overwrite:
        print(f"{output} already exists; skipping DreaMS encoding.")
        return

    data_dir = ROOT / "data" / dataset_name
    spectra = np.load(data_dir / "spectra.npy")
    metadata = pd.read_csv(data_dir / "metadata.csv")
    loader = DataLoader(
        SpectraDataset(spectra, metadata),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Encoding {len(spectra):,} {dataset_name} spectra with DreaMS "
        f"on {device} (batch_size={batch_size}, workers={workers})."
    )
    print("Loading the frozen DreaMS checkpoint...")
    model = PreTrainedModel.from_name("DreaMS_embedding").model.to(device).eval()
    print("DreaMS checkpoint loaded; starting inference.")
    embeddings = []
    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc=f"DreaMS spectra ({dataset_name})",
            unit="batch",
            dynamic_ncols=True,
        ):
            embeddings.append(model(batch.to(device)).cpu().numpy())

    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.concatenate(embeddings, axis=0))
    print(f"Wrote {len(spectra):,} DreaMS embeddings to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("massspecgym", "spectraverse"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    encode(args.dataset, args.batch_size, args.workers, args.overwrite)


if __name__ == "__main__":
    main()
