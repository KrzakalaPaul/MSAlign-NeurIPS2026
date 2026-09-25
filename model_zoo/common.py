"""Small utilities shared by the independently reproduced baselines."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR


def optimizer_with_scheduler(parameters, lr, weight_decay, n_warmup_steps, n_max_steps):
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

    def schedule(step):
        if step < n_warmup_steps:
            return step / max(1, n_warmup_steps)
        progress = (step - n_warmup_steps) / max(1, n_max_steps - n_warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

    return [optimizer], [{"scheduler": LambdaLR(optimizer, schedule), "interval": "step"}]


class AlignmentMLP(nn.Module):
    """MLP convention used by the EmbCos reference implementation."""

    def __init__(self, d_in, d_hidden, d_shared, n_hidden_layers=1, dropout=0.0, layernorm=False, **_):
        super().__init__()
        layers = []
        width = d_in
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(width, d_hidden))
            if layernorm:
                layers.append(nn.LayerNorm(d_hidden))
            layers.extend((nn.GELU(), nn.Dropout(dropout)))
            width = d_hidden
        layers.append(nn.Linear(width, d_shared))
        self.layers = nn.Sequential(*layers)

    def forward(self, values):
        return self.layers(values)


def batch_infonce(x, y, temperature=0.07):
    logits = x @ y.T / temperature
    labels = torch.arange(len(x), device=x.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    return loss, (logits.argmax(dim=1) == labels).float().mean().item()


def candidate_infonce(query, candidates, mask, temperature=0.07):
    logits = torch.einsum("bd,bkd->bk", query, candidates) / temperature
    logits = logits.masked_fill(~mask, float("-inf"))
    loss = F.cross_entropy(logits, torch.zeros(len(query), dtype=torch.long, device=query.device))
    return loss, (logits.argmax(dim=1) == 0).float().mean().item()


def keep_only_k_candidates(candidates, mask, k):
    """Subsample negatives while retaining the target at index zero."""

    negatives = torch.where(mask[1:])[0] + 1
    indices = torch.cat((torch.zeros(1, dtype=torch.long), negatives[torch.randperm(len(negatives))[: k - 1]]))
    selected, selected_mask = candidates[indices], mask[indices]
    if len(indices) < k:
        padding = k - len(indices)
        selected = torch.cat((selected, selected.new_zeros((padding, *selected.shape[1:]))))
        selected_mask = torch.cat((selected_mask, torch.zeros(padding, dtype=torch.bool)))
    return selected, selected_mask


def collate_candidates(candidates, masks):
    width = max(len(values) for values in candidates)
    shape = (len(candidates), width, *candidates[0].shape[1:])
    padded = candidates[0].new_zeros(shape)
    padded_mask = torch.zeros(len(candidates), width, dtype=torch.bool)
    for row, (values, mask) in enumerate(zip(candidates, masks)):
        padded[row, : len(values)] = values
        padded_mask[row, : len(mask)] = mask
    return padded, padded_mask
