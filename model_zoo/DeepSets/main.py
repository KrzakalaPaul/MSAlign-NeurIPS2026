"""Training entry point for the DeepSet retrieval baseline."""

from __future__ import annotations

from .training import train_and_eval_fingerprint_regression

from .model import DeepSet


def train_and_eval_DeepSet(args, config: dict):
    spectrum = config["spectrum"]
    model = DeepSet(config)
    return train_and_eval_fingerprint_regression(
        args,
        config,
        model=model,
        model_name="DeepSet",
        spectrum_transform="peaks",
        spectrum_options={
            "n_peaks": int(spectrum["n_peaks"]),
            "mz_min": float(spectrum["mz_min"]),
            "mz_max": float(spectrum["mz_max"]),
            "precursor_intensity": float(spectrum["precursor_intensity"]),
        },
    )
