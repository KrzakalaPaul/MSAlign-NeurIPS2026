#!/usr/bin/env bash
# Run this once on a login node. Do not submit it with sbatch.

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required on the login node." >&2
  exit 1
fi

echo "[1/2] Creating the main preprocessing/training environment"
uv sync --locked --extra preprocessing

echo "[2/2] Creating the isolated DreaMS environment"
uv sync --project playground --locked

test -x .venv/bin/python
test -x playground/.venv/bin/python
echo "Shared environments are ready. The SLURM jobs do not require uv."
