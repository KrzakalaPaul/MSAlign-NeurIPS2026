"""Compute raw or random-split-normalized sliced Wasserstein distance."""

from __future__ import annotations

import argparse
from pathlib import Path

from .distance import compute_swd
from .results import OUTPUT_PATH, get_result, normalized_swd, write_result


RANDOM_SPLITS = tuple(f"random_seed{seed}" for seed in range(5))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--ms-representation", required=True)
    parser.add_argument("--mol-representation", required=True)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    return parser.parse_args()


def main():
    args = parse_args()

    def obtain(split: str) -> float:
        cached = get_result(
            args.dataset,
            split,
            args.ms_representation,
            args.mol_representation,
        )
        if cached is not None:
            print(f"Using cached {args.dataset}/{split}: {cached:.8f}")
            return cached
        print(f"Computing {args.dataset}/{split}...", flush=True)
        value = compute_swd(
            dataset=args.dataset,
            split=split,
            ms_representation=args.ms_representation,
            mol_representation=args.mol_representation,
            batch_size=args.batch_size,
            n_workers=args.workers,
            data_root=args.data_root,
        )
        write_result(
            args.dataset,
            split,
            args.ms_representation,
            args.mol_representation,
            value,
        )
        print(f"Sliced Wasserstein-1 ({split}): {value:.8f}")
        return value

    distance = obtain(args.split)
    if args.normalize:
        references = [obtain(split) for split in RANDOM_SPLITS]
        normalized = normalized_swd(distance, references)
        print(f"Mean random-split SWD: {sum(references) / len(references):.8f}")
        print(f"Normalized SWD ({args.split}): {normalized:.8f}")
    print(f"Updated {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
