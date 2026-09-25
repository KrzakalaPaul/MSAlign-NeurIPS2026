"""Aggregate, calibrate, and evaluate modular candidate-score channels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from models.retrieval import retrieval_metrics_from_scores

from .cache import ScoreCache
from .scorer import Scorer, get_scorer
from .solver import get_solver


def _validate_scores(
    candidates_scores: torch.Tensor,
    candidates_masks: torch.Tensor,
    n_scorers: int | None = None,
) -> None:
    if candidates_scores.ndim != 3 or not candidates_scores.is_floating_point():
        raise ValueError("candidates_scores must be floating point with shape [N, K, S].")
    if candidates_masks.ndim != 2 or candidates_masks.dtype != torch.bool:
        raise ValueError("candidates_masks must be boolean with shape [N, K].")
    if candidates_scores.shape[:2] != candidates_masks.shape:
        raise ValueError("Score and mask candidate dimensions differ.")
    if candidates_scores.shape[0] == 0 or candidates_scores.shape[1] == 0:
        raise ValueError("Score tensors must contain samples and candidates.")
    if n_scorers is not None and candidates_scores.shape[2] != n_scorers:
        raise ValueError(
            f"Expected {n_scorers} score channels, got {candidates_scores.shape[2]}."
        )
    if not torch.all(candidates_masks[:, 0]):
        raise ValueError("Candidate zero must be valid for every sample.")
    valid_values = candidates_scores[candidates_masks]
    if not torch.all(torch.isfinite(valid_values)):
        raise ValueError("Every valid candidate score must be finite.")


def retrieval_metrics(
    combined_scores: torch.Tensor,
    candidates_masks: torch.Tensor,
    recalls: tuple[int, ...] = (1, 5, 20),
) -> dict[str, torch.Tensor]:
    """Recall metrics with score ties treated as failures at the boundary."""

    return retrieval_metrics_from_scores(combined_scores, candidates_masks, recalls)


class MSAlignFusion:
    """Combine fixed candidate-score channels with globally fitted weights."""

    def __init__(
        self,
        scorers_params_list: Sequence[Mapping[str, object]],
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        if not scorers_params_list:
            raise ValueError("MSAlignFusion requires at least one scorer.")
        self.scorers: list[Scorer] = [
            get_scorer(params) for params in scorers_params_list
        ]
        self.scorer_names = [scorer.name for scorer in self.scorers]
        if len(self.scorer_names) != len(set(self.scorer_names)):
            raise ValueError("Scorer names must be unique.")
        self.score_cache = ScoreCache(cache_dir) if cache_dir is not None else None
        self.alpha: torch.Tensor | None = None
        self.selected_temperature: float | None = None

    def get_scores(
        self,
        datasets_params: Mapping[str, object],
        fold: str,
        *,
        verbose: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores_list = []
        reference_mask = None
        for scorer in self.scorers:
            if verbose:
                print(f"  [{fold}] scoring with {scorer.name} ({scorer.scorer_type})", flush=True)
            compute = lambda scorer=scorer: scorer.get_score(
                datasets_params, fold, verbose=verbose
            )
            if self.score_cache is None:
                scores, mask = compute()
            else:
                scores, mask = self.score_cache.get_or_compute(
                    scorer,
                    datasets_params,
                    fold,
                    compute,
                    verbose=verbose,
                )
            if scores.ndim != 2 or scores.shape != mask.shape:
                raise ValueError("Each scorer must return [N, K] scores and masks.")
            if reference_mask is None:
                reference_mask = mask
            elif not torch.equal(reference_mask, mask):
                raise ValueError("All scorers must return identical candidate masks.")
            scores_list.append(scores)

        candidates_scores = torch.stack(scores_list, dim=-1)
        _validate_scores(candidates_scores, reference_mask, len(self.scorers))
        return candidates_scores, reference_mask

    def fit(
        self,
        candidates_scores: torch.Tensor,
        candidates_masks: torch.Tensor,
        fit_params: Mapping[str, object],
    ) -> torch.Tensor:
        _validate_scores(candidates_scores, candidates_masks, len(self.scorers))
        delta = candidates_scores[:, :1, :] - candidates_scores[:, 1:, :]
        delta_masks = candidates_masks[:, 1:]
        solver = get_solver(fit_params)
        self.alpha = solver(delta, delta_masks).detach()
        self.selected_temperature = getattr(solver, "selected_temperature", None)
        return self.alpha

    def eval(
        self,
        candidates_scores: torch.Tensor,
        candidates_masks: torch.Tensor,
    ) -> dict[str, object]:
        if self.alpha is None:
            raise RuntimeError("fit must be called before eval.")
        _validate_scores(candidates_scores, candidates_masks, len(self.scorers))
        alpha = self.alpha.to(
            device=candidates_scores.device, dtype=candidates_scores.dtype
        )
        combined_scores = (candidates_scores * alpha).sum(dim=-1)
        masked_scores = combined_scores.masked_fill(~candidates_masks, float("-inf"))
        # Put candidate zero last in the stable tie order so the explicit
        # ranking follows the same conservative policy as the R@K metrics.
        tie_order = torch.cat(
            (
                torch.arange(1, masked_scores.shape[1], device=masked_scores.device),
                torch.zeros(1, dtype=torch.long, device=masked_scores.device),
            )
        )
        positions = torch.argsort(
            masked_scores[:, tie_order], dim=-1, descending=True, stable=True
        )
        ranking = tie_order[positions]
        return {
            "alpha": alpha,
            "combined_scores": masked_scores,
            "ranking": ranking,
            "metrics": retrieval_metrics(masked_scores, candidates_masks),
        }
