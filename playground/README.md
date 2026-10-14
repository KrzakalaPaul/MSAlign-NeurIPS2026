# Raw-input playground

The playground runs a released MSAlign model directly from a centroided peak
list and candidate SMILES. It does not need a dataset, candidate map, or any
precomputed representations.

## Installation

The environment is isolated because DreaMS pins an older dependency stack.
Before the first run, accept the terms for
[`SaeedLab/MolDeBERTa-base-123M-mtr`](https://huggingface.co/SaeedLab/MolDeBERTa-base-123M-mtr).
A CUDA GPU is required. From the repository root:

```bash
uv sync --project playground --locked
export HF_TOKEN=hf_your_token_here
code .
```

Open `playground/MSAlign_playground.ipynb` directly in VS Code and select
`playground/.venv/bin/python` as the notebook kernel. No JupyterLab server is
required. Starting VS Code from this terminal passes `HF_TOKEN` to the kernel.

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

## Input format and preprocessing

### Spectrum

The caller must provide a **centroided MS/MS spectrum**. `peaks` is an
`(n_peaks, 2)` array-like object whose rows are `[m/z, intensity]`, ordered by
increasing `m/z`. Values must be finite and non-negative, and at least one
intensity must be positive. Intensities may use any common scale: the model
normalizes them internally.

Centroiding, deisotoping, noise filtering, and merging duplicate peaks are not
performed by the playground. Apply those operations beforehand if they are
needed for your instrument output. Pass fragment peaks only: do **not** add the
DreaMS precursor pseudo-peak yourself.

`precursor_mz` is the measured ion mass-to-charge ratio in Thomson, not the
neutral molecular mass. It must be positive and finite. The playground applies
the same two spectrum transforms used during training:

- **DreaMS:** retain the 100 most intense fragment peaks, pad shorter peak
  lists, scale intensities relative to the base peak, and prepend
  `[precursor_mz, 1.1]` before running the frozen DreaMS encoder.
- **Binned spectrum:** sum intensities into 0.1-Th bins up to 1005 Th, scale the
  largest bin to 999, then apply `log10(x + 1) / 3`.

The raw MSAlign release uses only DreaMS. MSAlign+ computes both transforms once
and reuses them across its scorers.

### Metadata

`adduct` is an optional string such as `[M+H]+`, `[M+Na]+`, or `[M-H]-`.
Whitespace is removed, and the common aliases `[M+CH3COO]-`, `[M+FA-H]-`, and
`[M+HCOO]-` are normalized. A missing or unrecognized value uses the alignment
model's learned unknown-adduct embedding. It also disables the mass-score
channels in MSAlign+, because a neutral mass cannot be inferred reliably.

`collision_energy` is optional. Supply a finite numerical value in the
convention reported by the experiment; the playground does not convert between
eV, normalized collision energy, or instrument-specific conventions. Both
`None` and NaN use the learned missing-energy embedding.

### Candidate molecules

Each candidate must be a valid, RDKit-parseable SMILES for the neutral candidate
molecule—not an adducted ion or a molecular formula. Salt removal, parent-fragment
selection, and charge normalization are not performed, so supply the molecular
form you intend to retrieve. Internally, the playground:

1. parses and sanitizes the SMILES with RDKit;
2. converts it to canonical isomeric SMILES;
3. computes its MolDeBERTa embedding and, for MSAlign+, its radius-2 4096-bit
   Morgan fingerprint and RDKit exact molecular mass.

MolDeBERTa inputs longer than its supported sequence length are truncated. The
playground does not deduplicate candidates or apply molecular-formula or mass
filters. A list is scored in the supplied order.

In short, users provide a clean centroided peak list, its precursor and optional
experimental metadata, plus chemically standardized candidate SMILES. Neural
embeddings, fixed vectorizations, metadata encoding, and score fusion happen
inside `MSAlign`.

The same example is available in
[`MSAlign_playground.ipynb`](MSAlign_playground.ipynb).
