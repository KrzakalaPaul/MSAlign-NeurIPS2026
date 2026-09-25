# MVP

This implementation reproduces MVP's four-view training objective (molecular
graph, fingerprint, individual spectrum, and consensus spectrum). Retrieval is
reported from the molecule–individual-spectrum view, as in the MSAlign paper.

The released configuration uses formula-annotated peaks. It therefore requires
the target molecular formula and is explicitly reported as oracle-assisted.

```bash
python -m model_zoo.train mvp \
  --dataset massspecgym \
  --candidate-map official_candidates_by_mass \
  --split formula_seed1
```
