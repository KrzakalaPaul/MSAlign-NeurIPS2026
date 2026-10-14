#!/usr/bin/env bash
#SBATCH --job-name=msalign-release-encode
#SBATCH --partition=V100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --export=ALL

set -euo pipefail

REPOSITORY_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPOSITORY_ROOT"
test -x .venv/bin/python || { echo "Run scripts/00_setup_login.sh first." >&2; exit 1; }
test -x playground/.venv/bin/python || { echo "Run scripts/00_setup_login.sh first." >&2; exit 1; }
test -f data/massspecgym/candidates/official_candidates_by_mass/map.json

source .venv/bin/activate
export MSALIGN_DREAMS_PYTHON="$REPOSITORY_ROOT/playground/.venv/bin/python"
CUDA_LIBRARY_PATHS=$(python -c 'import site, pathlib; print(":".join(str(path) for root in site.getsitepackages() for path in pathlib.Path(root, "nvidia").glob("*/lib") if path.exists()))')
export LD_LIBRARY_PATH="${CUDA_LIBRARY_PATHS}:${LD_LIBRARY_PATH:-}"

echo "Commit: $(git rev-parse HEAD)"
srun python -u precompute_representations.py massspecgym \
  --candidate-map official_candidates_by_mass \
  --representations dreams moldeberta fingerprint \
  --batch-size 32 \
  --workers "${SLURM_CPUS_PER_TASK:-16}"

test -f data/massspecgym/spectra_embeddings/dreams.npy
test -f data/massspecgym/candidates/official_candidates_by_mass/moldeberta_base_123m_mtr.h5
test -f data/massspecgym/candidates/official_candidates_by_mass/morgan_2_4096.h5
echo "All released input representations are ready."
