#!/usr/bin/env bash
#SBATCH --job-name=msalign-release-data
#SBATCH --partition=CPU
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --export=ALL

set -euo pipefail

REPOSITORY_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPOSITORY_ROOT"
test -x .venv/bin/python || { echo "Run scripts/00_setup_login.sh first." >&2; exit 1; }
source .venv/bin/activate

echo "Commit: $(git rev-parse HEAD)"
echo "Downloading the prepared MassSpecGym release from Zenodo"
srun python -u download_prepared_data.py massspecgym
test -f data/massspecgym/candidates/official_candidates_by_mass/map.json
test -f data/massspecgym/splits/formula_seed0.csv
test -f data/massspecgym/splits/mces_1.csv
echo "Prepared MassSpecGym data are ready."
