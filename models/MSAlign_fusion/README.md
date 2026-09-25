# Score fusion

MSAlign⁴ combines the candidate scores of models trained with the four pairs in
`{DreaMS, bins} × {MolDeBERTa, Morgan}`. MSAlign⁴M additionally includes three
Gaussian mass scores with standard deviations 0.1, 1, and 10 ppm.

For weights on the simplex, calibration minimizes a convex listwise loss. It
does so for a log-spaced grid of temperatures and retains the weights with the
best strict validation R@1. The selected weights are then evaluated once on the
test fold. Candidate-score tensors are cached, so changing the calibration grid
does not rerun the alignment models.

```bash
python fuse_and_eval.py --config models/MSAlign_fusion/configs/default.yaml --verbose
```

Setting `formula_filter: true` applies the oracle molecular-formula filter to
both validation calibration and test evaluation. It is disabled by default.
