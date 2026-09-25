"""Canonicalize molecules, create a split, and construct candidate sets."""

import argparse

from preprocessing import (
    download_massspecgym_official_candidate_map,
    prepare_candidates,
    process_smiles,
    split,
)
from preprocessing.create_mces_splits import create_mces_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("massspecgym", "spectraverse"))
    parser.add_argument("--split", required=True, choices=("formula", "mces"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print(f"Preparing {args.dataset} (split={args.split}, seed={args.seed}).")
    print("\n[1/3] Canonicalizing and indexing molecules")
    process_smiles(args.dataset, overwrite=args.overwrite, n_threads=args.workers)
    print("\n[2/3] Creating dataset split")
    if args.split == "formula":
        split(args.dataset, "formula", overwrite=args.overwrite, seed=args.seed, add_seed_name=True)
    else:
        if args.dataset != "massspecgym":
            raise ValueError("MCES splits are only defined for MassSpecGym")
        split(args.dataset, "as_provided", overwrite=args.overwrite)
        create_mces_splits(overwrite=args.overwrite)
    print("\n[3/3] Preparing retrieval candidates")
    if args.dataset == "massspecgym":
        download_massspecgym_official_candidate_map(
            overwrite=args.overwrite, n_threads=args.workers, kind="mass"
        )
    else:
        prepare_candidates(
            args.dataset, n_candidates=args.candidates, kind="mass",
            overwrite=args.overwrite, seed=args.seed, n_threads=args.workers,
        )
    print(f"\nFinished preparing {args.dataset}.")


if __name__ == "__main__":
    main()
