# Raw-input playground

The playground runs a released MSAlign model directly from a centroided peak
list and candidate SMILES. It does not need a dataset, candidate map, or any
precomputed representations.

## Installation

The environment is isolated because DreaMS pins an older dependency stack.
From the repository root:

```bash
uv sync --project playground --locked
source playground/.venv/bin/activate
jupyter lab playground/MSAlign_playground.ipynb
```

A CUDA GPU is required. Before the first run, accept the terms for
[`SaeedLab/MolDeBERTa-base-123M-mtr`](https://huggingface.co/SaeedLab/MolDeBERTa-base-123M-mtr)
and export a Hugging Face token:

```bash
export HF_TOKEN=hf_your_token_here
```

DreaMS and MolDeBERTa are downloaded from their official sources on first use
and then read from their normal local caches. Their weights are not included in
the MSAlign checkpoint archives.

## Checkpoint archives

Extract one downloaded archive directly under `checkpoints/`:

```bash
unzip msalign_massspecgym_formula_seed0.zip -d checkpoints
```

Both raw MSAlign and fused MSAlign+ archives expose the same interface. The
loader reads `manifest.yaml`, verifies the model files, and selects the correct
scorers automatically.

The four release names are
`msalign_massspecgym_formula_seed0`,
`msalign_star_massspecgym_formula_seed0`,
`msalign_massspecgym_mces_1`, and
`msalign_star_massspecgym_mces_1`.

## Python interface

```python
from playground import MSAlign

model = MSAlign("checkpoints/msalign_massspecgym_formula_seed0")

spectrum = {
    "peaks": [
        [77.0390, 0.12],
        [105.0335, 0.46],
        [121.0284, 1.00],
        [138.0550, 0.31],
    ],
    "precursor_mz": 181.0495,
    "adduct": "[M+H]+",       # optional
    "collision_energy": 30.0, # optional
}

score = model(spectrum, "CC(=O)Oc1ccccc1C(=O)O")
scores = model(spectrum, ["CCO", "CCCO", "c1ccccc1"])
```

`peaks` and `precursor_mz` are required. `adduct` and `collision_energy` may be
omitted or set to `None`. A string candidate returns a `float`; a list returns a
list of scores in the same order. These values are relative retrieval scores,
not calibrated probabilities.

The same example is available in
[`MSAlign_playground.ipynb`](MSAlign_playground.ipynb).
