#!/usr/bin/env bash
# Run this once on a login node. Do not submit it with sbatch.

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required on the login node." >&2
  exit 1
fi

# Installing the CUDA wheels in parallel can exceed per-process limits on
# shared login nodes. Users may override these conservative defaults.
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-2}"
export UV_CONCURRENT_BUILDS="${UV_CONCURRENT_BUILDS:-1}"
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-1}"

echo "[1/2] Creating the main preprocessing/training environment"
uv sync --locked --extra preprocessing

echo "[2/2] Creating the isolated DreaMS environment"
uv sync --project playground --locked

test -x .venv/bin/python
test -x playground/.venv/bin/python
echo "Shared environments are ready. The SLURM jobs do not require uv."
