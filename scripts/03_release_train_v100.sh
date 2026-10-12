#!/usr/bin/env bash
#SBATCH --job-name=msalign-release-train
#SBATCH --partition=V100
#SBATCH --array=0-3
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --export=ALL

set -euo pipefail

REPOSITORY_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPOSITORY_ROOT"
test -x .venv/bin/python || { echo "Run scripts/00_setup_login.sh first." >&2; exit 1; }
test -f data/massspecgym/spectra_embeddings/dreams.npy
source .venv/bin/activate

CUDA_LIBRARY_PATHS=$(python -c 'import site, pathlib; print(":".join(str(path) for root in site.getsitepackages() for path in pathlib.Path(root, "nvidia").glob("*/lib") if path.exists()))')
export LD_LIBRARY_PATH="${CUDA_LIBRARY_PATHS}:${LD_LIBRARY_PATH:-}"

SPECTRUM_REPRESENTATIONS=(dreams dreams bins bins)
MOLECULE_REPRESENTATIONS=(moldeberta_base_123m_mtr morgan_2_4096 moldeberta_base_123m_mtr morgan_2_4096)
NAMES=(dreams_moldeberta dreams_fingerprint bins_moldeberta bins_fingerprint)

TASK=${SLURM_ARRAY_TASK_ID:?Submit this script as the declared array}
SPECTRUM=${SPECTRUM_REPRESENTATIONS[$TASK]}
MOLECULE=${MOLECULE_REPRESENTATIONS[$TASK]}
NAME=${NAMES[$TASK]}

train_one() {
  local split=$1
  local config=$2
  local checkpoint="checkpoints/${NAME}/massspecgym__official_candidates_by_mass__${split}.ckpt"
  if [[ -f "$checkpoint" ]]; then
    echo "Skipping completed checkpoint: $checkpoint"
    return
  fi
  srun python -u train_msalign.py \
    --config "$config" \
    --dataset massspecgym \
    --candidate-map official_candidates_by_mass \
    --split "$split" \
    --spectrum-representation "$SPECTRUM" \
    --molecule-representation "$MOLECULE" \
    --output-checkpoint "$checkpoint" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --test-batch-size 16 \
    --no-logger
  test -f "$checkpoint"
}

echo "Commit: $(git rev-parse HEAD)"
echo "Training representation pair: $SPECTRUM / $MOLECULE"
train_one formula_seed0 massspecgym_formula
train_one mces_1 massspecgym_mces
echo "Finished $NAME."
