"""Four-view MultiView Projection model."""

from __future__ import annotations

from itertools import combinations

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from models.retrieval import retrieval_metrics_from_scores
from .encoders import (
    BinnedSpectrumEncoder,
    FingerprintEncoder,
    FormulaSpectrumEncoder,
    molecular_graph_encoder,
)


def symmetric_inbatch_infonce(first: torch.Tensor, second: torch.Tensor, temperature: float):
    first, second = F.normalize(first, dim=-1), F.normalize(second, dim=-1)
    logits = first @ second.T / temperature
    targets = torch.arange(len(first), device=first.device)
    return F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)


class MVP(pl.LightningModule):
    """Align graph, fingerprint, spectrum, and consensus-spectrum views."""

    VIEW_NAMES = ("molecule", "fingerprint", "spectrum", "consensus")

    def __init__(self, config: dict):
        super().__init__()
        if config["spectrum_mode"] not in {"formula", "bins"}:
            raise ValueError("spectrum_mode must be 'formula' or 'bins'.")
        self.config = config
        d_shared = config["d_shared"]
        self.molecule_encoder = molecular_graph_encoder(config)
        self.fingerprint_encoder = FingerprintEncoder(1024, d_shared)
        if config["spectrum_mode"] == "formula":
            arguments = dict(
                d_in=15, d_out=d_shared,
                formula_dimensions=config["formula_encoder"]["dimensions"],
                formula_dropout=config["formula_encoder"]["dropout"],
                n_transformer_layers=config["formula_encoder"]["n_transformer_layers"],
                n_heads=config["formula_encoder"]["n_heads"],
            )
            self.spectrum_encoder = FormulaSpectrumEncoder(**arguments)
            self.consensus_encoder = FormulaSpectrumEncoder(**arguments)
        else:
            d_bins = int(torch.ceil(torch.tensor(config["spectrum"]["max_mz"] / config["spectrum"]["bin_width"])).item())
            self.spectrum_encoder = BinnedSpectrumEncoder(d_bins, d_shared, config["fc_dropout"])
            self.consensus_encoder = BinnedSpectrumEncoder(d_bins, d_shared, config["fc_dropout"])
        self.temperature = float(config["temperature"])
        self.lr = float(config["optimization"]["lr"])
        self.weight_decay = float(config["optimization"]["weight_decay"])
        self.save_hyperparameters(config)

    def _encode_spectrum(self, encoder, value):
        if self.config["spectrum_mode"] == "formula":
            return encoder(value["tokens"], value["padding_mask"])
        return encoder(value)

    def encode_views(self, batch):
        return {
            "molecule": self.molecule_encoder(batch["molecule"]),
            "fingerprint": self.fingerprint_encoder(batch["fingerprint"]),
            "spectrum": self._encode_spectrum(self.spectrum_encoder, batch["spectrum"]),
            "consensus": self._encode_spectrum(self.consensus_encoder, batch["consensus"]),
        }

    def contrastive_loss(self, batch):
        views = self.encode_views(batch)
        losses = {
            f"{first}_{second}": symmetric_inbatch_infonce(
                views[first], views[second], self.temperature
            )
            for first, second in combinations(self.VIEW_NAMES, 2)
        }
        return sum(losses.values()), losses

    def _contrastive_step(self, batch, stage):
        loss, pair_losses = self.contrastive_loss(batch)
        self.log(f"loss ({stage})", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch["fingerprint"]))
        self.log_dict({f"loss {name} ({stage})": value for name, value in pair_losses.items()}, on_step=False, on_epoch=True, batch_size=len(batch["fingerprint"]))
        return loss

    def training_step(self, batch, batch_idx=0): return self._contrastive_step(batch, "train")
    def validation_step(self, batch, batch_idx=0): return self._contrastive_step(batch, "val")

    def test_step(self, batch, batch_idx=0):
        spectrum, candidate_graphs = batch
        spectrum = F.normalize(self._encode_spectrum(self.spectrum_encoder, spectrum), dim=-1)
        maximum = max(graph.batch_size for graph in candidate_graphs)
        scores = spectrum.new_full((len(candidate_graphs), maximum), float("-inf"))
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for row, graph in enumerate(candidate_graphs):
            molecules = F.normalize(self.molecule_encoder(graph), dim=-1)
            scores[row, :len(molecules)] = spectrum[row] @ molecules.T
            mask[row, :len(molecules)] = True
        metrics = retrieval_metrics_from_scores(scores, mask)
        self.log_dict({f"{name} (test)": value for name, value in metrics.items()}, on_step=False, on_epoch=True, batch_size=len(candidate_graphs))
        return metrics["R@1"]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
