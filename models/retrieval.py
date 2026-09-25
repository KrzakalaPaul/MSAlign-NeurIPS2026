"""Shared retrieval helpers for the independently reproduced baselines."""

from __future__ import annotations

import torch

from models.MSAlign.losses import candidate_retrieval_metrics


retrieval_metrics_from_scores = candidate_retrieval_metrics


def candidate_retrieval_accuracy(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    candidate_mask: torch.Tensor,
    recalls: tuple[int, ...] = (1, 5, 20),
) -> dict[str, float]:
    scores = torch.einsum("nd,nkd->nk", queries, candidates)
    return {
        name: float(value)
        for name, value in candidate_retrieval_metrics(scores, candidate_mask, recalls).items()
    }
