"""Train one MSAlign model.

Example::

    python train_msalign.py --config massspecgym_formula --dataset massspecgym \
        --candidate-map official_candidates_by_mass --split formula_seed1

The representation flags are the only architecture switch needed to train the
four constituents of MSAlign+.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from models.MSAlign.main import train_MSAlign


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("massspecgym_formula", "massspecgym_mces", "spectraverse_formula"), required=True)
    parser.add_argument("--dataset", dest="labelled_dataset_name", choices=("massspecgym", "spectraverse"), required=True)
    parser.add_argument("--candidate-map", dest="candidate_map_name", required=True)
    parser.add_argument("--split", dest="split_method", required=True)
    parser.add_argument("--spectrum-representation", choices=("dreams", "bins"))
    parser.add_argument("--molecule-representation", choices=("moldeberta_base_123m_mtr", "morgan_2_4096"))
    parser.add_argument("--all-representations", action="store_true", help="Train all four models used by MSAlign+")
    parser.add_argument("--output-checkpoint")
    parser.add_argument("--workers", dest="n_workers", type=int, default=8)
    parser.add_argument("--test-batch-size", dest="batch_size_test", type=int, default=16)
    parser.add_argument("--wandb-project", default="MSAlign")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--no-logger", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path("models/MSAlign/configs") / f"{args.config}.yaml"
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not args.all_representations:
        if args.spectrum_representation:
            config["representations"]["spectrum"] = args.spectrum_representation
        if args.molecule_representation:
            config["representations"]["molecule"] = args.molecule_representation
        train_MSAlign(args, config)
        return

    if args.spectrum_representation or args.molecule_representation or args.output_checkpoint:
        raise ValueError("--all-representations cannot be combined with representation or output overrides")
    pairs = (
        ("dreams", "moldeberta_base_123m_mtr", "dreams_moldeberta"),
        ("dreams", "morgan_2_4096", "dreams_fingerprint"),
        ("bins", "moldeberta_base_123m_mtr", "bins_moldeberta"),
        ("bins", "morgan_2_4096", "bins_fingerprint"),
    )
    for spectrum, molecule, name in pairs:
        pair_config = copy.deepcopy(config)
        pair_config["representations"] = {"spectrum": spectrum, "molecule": molecule}
        pair_args = copy.copy(args)
        filename = "__".join((args.labelled_dataset_name, args.candidate_map_name, args.split_method)) + ".ckpt"
        pair_args.output_checkpoint = str(Path("checkpoints") / name / filename)
        pair_args.wandb_run_name = f"{name}/{args.labelled_dataset_name}/{args.split_method}"
        if Path(pair_args.output_checkpoint).is_file():
            print(f"Skipping completed checkpoint: {pair_args.output_checkpoint}")
            continue
        train_MSAlign(pair_args, pair_config)


if __name__ == "__main__":
    main()
