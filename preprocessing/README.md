# Preprocessing

Install the optional preprocessing environment before rebuilding raw data or
frozen neural representations:

```bash
uv sync --extra preprocessing
```

This extra contains the raw-data and MolDeBERTa dependencies. It is unnecessary
when using the released precomputed representations. DreaMS is intentionally
not installed into the project environment: `precompute_representations.py`
launches `encode_dreams.py` through an isolated `uv` script environment because
the official package has legacy pins that conflict with modern RDKit.

The public pipeline has three stages:

1. `download_data.py` downloads the raw MassSpecGym or SpectraVerse files.
2. `prepare_data.py` canonicalizes SMILES, creates formula or MCES splits, and
   constructs the candidate map.
3. `precompute_representations.py` computes the frozen DreaMS, MolDeBERTa, and
   packed Morgan representations used by MSAlign⁴. Pass `--annotate-peaks` to
   also generate the peak-subformula annotations consumed by FLARE and MVP.

All artifacts are stored under `data/<dataset>/`. Candidate representations use
the row layout `[unique molecule, candidate position, feature]` and share a
boolean candidate mask. Position zero must always contain the target.

MolDeBERTa requires a GPU and an `HF_TOKEN`. Morgan fingerprints are computed
on CPU. DreaMS uses its frozen public checkpoint.

Processed artifacts can instead be downloaded from the Zenodo archives listed
in [`DATA.md`](../DATA.md).
