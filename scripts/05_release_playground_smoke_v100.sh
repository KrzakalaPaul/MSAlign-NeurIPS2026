#!/usr/bin/env bash
#SBATCH --job-name=msalign-release-smoke
#SBATCH --partition=V100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --export=ALL

set -euo pipefail

REPOSITORY_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPOSITORY_ROOT"
test -x playground/.venv/bin/python || { echo "Run scripts/00_setup_login.sh first." >&2; exit 1; }

TEMPORARY=$(mktemp -d)
trap 'rm -rf "$TEMPORARY"' EXIT

for archive in releases/msalign_*.zip; do
  playground/.venv/bin/python -m zipfile -e "$archive" "$TEMPORARY"
done

for bundle in "$TEMPORARY"/msalign_*; do
  if [[ "$(basename "$bundle")" == *mces_1 ]]; then
    SPLIT=mces_1
  else
    SPLIT=formula_seed0
  fi
  srun playground/.venv/bin/python -u -m scripts.verify_playground_representations \
    "$bundle" --split "$SPLIT"
  (
    cd "$TEMPORARY"
    srun env PYTHONPATH="$REPOSITORY_ROOT" \
      "$REPOSITORY_ROOT/playground/.venv/bin/python" -u -m playground.smoke "$bundle"
  )
done
echo "All four portable bundles passed raw-input inference."
