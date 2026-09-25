# Alignment model

MSAlign maps one spectrum representation and one molecular representation to a
shared space. The released model uses DreaMS and MolDeBERTa; changing the two
representation names also trains the four constituents used by MSAlign⁴.

```text
spectrum -> MLP -> concatenate(energy, adduct) -> MLP -> normalize
molecule ---------------------------------------> MLP -> normalize
```

Candidates are scored by temperature-scaled cosine similarity. During
training, candidate zero is the positive and the remaining sampled candidates
are negatives in a cross-entropy loss. At evaluation, all candidates are used;
invalid padding is assigned `-inf`, and a score tie with a negative counts as a
failure.

The three released configurations are `massspecgym_formula`,
`massspecgym_mces`, and `spectraverse_formula`. For example:

```bash
python train_msalign.py \
  --config massspecgym_formula \
  --dataset massspecgym \
  --candidate-map official_candidates_by_mass \
  --split formula_seed1 \
  --output-checkpoint checkpoints/dreams_moldeberta/massspecgym__official_candidates_by_mass__formula_seed1.ckpt

python eval.py \
  --checkpoint checkpoints/dreams_moldeberta/massspecgym__official_candidates_by_mass__formula_seed1.ckpt \
  --dataset massspecgym \
  --candidate-map official_candidates_by_mass \
  --split formula_seed1
```
