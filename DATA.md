# Data release inventory

The processed datasets are published in
[Zenodo record 22830464](https://zenodo.org/records/22830464). Install either
archive with `python download_prepared_data.py <dataset>`.

## Archive 1: processed datasets

For MassSpecGym and SpectraVerse:

- `metadata.csv`, `unique_smiles.csv`, and the processed spectrum array;
- the five formula splits used in the paper;
- MassSpecGym `mces_1` and `mces_2`;
- the candidate maps used by each benchmark;
- peak subformula annotations required by FLARE and MVP;
- a manifest containing source URLs, versions, sizes, and SHA-256 checksums.

Frozen DreaMS, MolDeBERTa, and Morgan representations are generated locally by
`precompute_representations.py`; they are not part of the dataset archives.
`download_data.py` and `prepare_data.py` can instead rebuild the released
dataset layout from the original sources.

## Checkpoint archives

Four separate archives are prepared for raw MSAlign and MSAlign+ on
MassSpecGym `formula_seed0` and the official `mces_1` split. Each archive
contains compact MSAlign weights, a readable manifest, provenance, and
checksums. Foundation-model weights, competitor checkpoints, and fusion score
caches are not redistributed.
