# Reproduced baselines

This directory contains the five competitor implementations reported in the
MSAlign paper: DeepSets, EmbCos, JESTR, FLARE, and MVP. They use the same
processed datasets, candidate ordering, and strict-tie retrieval metrics as
MSAlign, while preserving each paper's architecture and optimization setup.

Train a baseline with:

```bash
python -m model_zoo.train embcos \
  --dataset massspecgym \
  --candidate-map official_candidates_by_mass \
  --split formula_seed1
```

The target molecule is candidate zero. At evaluation, padded candidates are
masked and an exact score tie with a negative counts against the target.

DeepSets and EmbCos run in the default MSAlign environment. The graph-based
JESTR, FLARE, and MVP implementations are isolated behind:

```bash
uv sync --extra graph-baselines
```

The released graph environment uses DGL 2.1 with CUDA 12.1, matching the paper
runs. It is never imported by MSAlign, score fusion, DeepSets, or EmbCos.
