# Checkpoint release pipeline

These temporary SLURM scripts test the public Zenodo workflow from a fresh
clone and produce the four playground archives. Run setup on the login node,
where `uv` is available:

```bash
bash scripts/00_setup_login.sh
export HF_TOKEN=hf_your_token_here
```

The remaining stages are:

1. `01_release_download_cpu.sh`: download and verify the prepared MassSpecGym
   archive from Zenodo.
2. `02_release_precompute_v100.sh`: compute DreaMS, MolDeBERTa, and Morgan
   representations.
3. `03_release_train_v100.sh`: four parallel V100 tasks, each training one
   representation pair on `formula_seed0` and `mces_1`.
4. `04_release_fuse_export_v100.sh`: calibrate on validation, evaluate on test,
   and create the four release ZIPs.
5. `05_release_playground_smoke_v100.sh`: compare scorer channels with the
   training-time caches, then run raw-input inference from a temporary working
   directory where no precomputed representations are visible.

Submit the chain from the repository root:

```bash
data_job=$(sbatch --parsable scripts/01_release_download_cpu.sh)
encode_job=$(sbatch --parsable --dependency="afterok:${data_job}" \
  scripts/02_release_precompute_v100.sh)
train_job=$(sbatch --parsable --dependency="afterok:${encode_job}" \
  scripts/03_release_train_v100.sh)
export_job=$(sbatch --parsable --dependency="afterok:${train_job}" \
  scripts/04_release_fuse_export_v100.sh)
sbatch --dependency="afterok:${export_job}" \
  scripts/05_release_playground_smoke_v100.sh
```

Every script records the exact Git commit. Existing data, representations,
checkpoints, score caches, and completed calibration CSVs are reused. Final
archives and their SHA-256 inventory are written under `releases/`.

Do not submit these jobs until the exact commit and requested resources have
been reviewed.
