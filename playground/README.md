# Raw-input playground

This isolated environment contains both DreaMS and MolDeBERTa. It is separate
from the main MSAlign environment because DreaMS pins a large legacy dependency
stack.

From the repository root:

```bash
uv sync --project playground
HF_TOKEN=hf_your_token_here uv run --project playground \
  jupyter lab playground/MSAlign_playground.ipynb
```

The Hugging Face account behind `HF_TOKEN` must have accepted the terms for
`SaeedLab/MolDeBERTa-base-123M-mtr`. The first execution also downloads the
frozen DreaMS and MolDeBERTa weights. A CUDA GPU is required.

Only checkpoints produced by the current code contain the ordered adduct
vocabulary required for standalone inference.
