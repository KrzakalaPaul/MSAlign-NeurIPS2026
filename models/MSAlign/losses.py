"""Candidate loss and retrieval metrics.

Candidate zero is always the target. Invalid padded candidates receive
``-inf``. Recall uses a strict inequality: a target tied with the K-th score is
counted as a failure, matching the conservative evaluation in the paper.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mask_logits(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if scores.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("scores and boolean candidate mask must have the same shape")
    if not torch.all(mask[:, 0]):
        raise ValueError("candidate zero must be valid for every spectrum")
    return scores.masked_fill(~mask, float("-inf"))


def candidate_cross_entropy(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    targets = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
    return F.cross_entropy(mask_logits(scores, mask), targets)


def candidate_retrieval_metrics(
    scores: torch.Tensor, mask: torch.Tensor, ks: tuple[int, ...] = (1, 5, 20)
) -> dict[str, torch.Tensor]:
    scores = mask_logits(scores, mask)
    target = scores[:, :1]
    negatives = scores[:, 1:]
    rank = 1 + ((negatives >= target) & mask[:, 1:]).sum(dim=1)
    return {f"R@{k}": (rank <= k).float().mean() for k in ks}
