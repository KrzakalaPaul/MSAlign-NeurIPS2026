#!/usr/bin/env bash
#SBATCH --job-name=msalign-release-fuse
#SBATCH --partition=V100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --export=ALL

set -euo pipefail

REPOSITORY_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPOSITORY_ROOT"
test -x .venv/bin/python || { echo "Run scripts/00_setup_login.sh first." >&2; exit 1; }
source .venv/bin/activate

CUDA_LIBRARY_PATHS=$(python -c 'import site, pathlib; print(":".join(str(path) for root in site.getsitepackages() for path in pathlib.Path(root, "nvidia").glob("*/lib") if path.exists()))')
export LD_LIBRARY_PATH="${CUDA_LIBRARY_PATHS}:${LD_LIBRARY_PATH:-}"

echo "Commit: $(git rev-parse HEAD)"

if [[ ! -f results/release_massspecgym_formula_seed0.csv ]]; then
  srun python -u fuse_and_eval.py \
    --config models/MSAlign_fusion/configs/release_massspecgym_formula_seed0.yaml \
    --verbose
fi
if [[ ! -f results/release_massspecgym_mces_1.csv ]]; then
  srun python -u fuse_and_eval.py \
    --config models/MSAlign_fusion/configs/release_massspecgym_mces_1.yaml \
    --verbose
fi

python -u export_checkpoint_bundles.py \
  --split formula_seed0 \
  --fusion-results results/release_massspecgym_formula_seed0.csv \
  --overwrite
python -u export_checkpoint_bundles.py \
  --split mces_1 \
  --fusion-results results/release_massspecgym_mces_1.csv \
  --overwrite

test -f releases/msalign_massspecgym_formula_seed0.zip
test -f releases/msalign_star_massspecgym_formula_seed0.zip
test -f releases/msalign_massspecgym_mces_1.zip
test -f releases/msalign_star_massspecgym_mces_1.zip
test -f releases/inventory.yaml
echo "Calibration and release packaging completed."
