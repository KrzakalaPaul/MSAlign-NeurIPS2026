"""Evaluate a raw MSAlign checkpoint on the test candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.MSAlign import candidate_retrieval_metrics
from models.MSAlign_fusion.scorer import MSAlignScorer, resolve_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=("massspecgym", "spectraverse"), required=True)
    parser.add_argument("--candidate-map", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = {
        "labelled_dataset_name": args.dataset,
        "candidate_map_name": args.candidate_map,
        "split_method": args.split,
    }
    checkpoint = resolve_checkpoint(args.checkpoint, dataset)
    scorer = MSAlignScorer(
        checkpoint,
        batch_size=args.batch_size,
        n_workers=args.workers,
        device=args.device,
    )
    scores, candidate_mask = scorer.get_score(dataset, fold="test", verbose=args.verbose)
    metrics = {
        name: float(value)
        for name, value in candidate_retrieval_metrics(scores, candidate_mask).items()
    }
    result = {
        "model": "MSAlign",
        "dataset": args.dataset,
        "candidate_map": args.candidate_map,
        "split": args.split,
        "fold": "test",
        "checkpoint": str(checkpoint),
        "n_test": scores.shape[0],
        **metrics,
    }

    print(f"Checkpoint: {checkpoint}")
    print(f"Test spectra: {result['n_test']:,}")
    print(", ".join(f"{name}={value:.4f}" for name, value in metrics.items()))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
