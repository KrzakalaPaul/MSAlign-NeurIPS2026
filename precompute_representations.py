"""Precompute the frozen representations used by MSAlign and its baselines.

DreaMS runs in a separate environment because its official package has strict,
legacy dependency pins. The isolation is transparent to this command and keeps
those packages out of the normal MSAlign environment.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from preprocessing import (
    annotate_peaks,
    get_molecule_embeddings,
    get_molecule_embeddings_for_candidates,
    get_molecule_fingerprint,
    get_molecule_fingerprint_for_candidates,
)


ROOT = Path(__file__).resolve().parent


def run_dreams(dataset, batch_size, workers, overwrite):
    output = ROOT / "data" / dataset / "spectra_embeddings" / "dreams.npy"
    if output.exists() and not overwrite:
        print(f"{output} already exists; skipping DreaMS encoding.", flush=True)
        return

    script = ROOT / "preprocessing" / "encode_dreams.py"
    configured_python = os.environ.get("MSALIGN_DREAMS_PYTHON")
    playground_python = ROOT / "playground" / ".venv" / "bin" / "python"
    dreams_python = Path(configured_python) if configured_python else playground_python
    if dreams_python.is_file():
        command = [str(dreams_python), str(script), dataset]
        runtime = str(dreams_python)
    else:
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError(
                "DreaMS is not available. Run scripts/00_setup_login.sh on a "
                "login node, or set MSALIGN_DREAMS_PYTHON to a prepared "
                "DreaMS Python interpreter."
            )
        command = [uv, "run", "--script", str(script), dataset]
        runtime = "an isolated uv environment"
    command.extend(
        ["--batch-size", str(batch_size), "--workers", str(workers)]
    )
    if overwrite:
        command.append("--overwrite")
    print(f"Launching DreaMS preprocessing with {runtime}...", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    print("DreaMS preprocessing finished.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("massspecgym", "spectraverse"))
    parser.add_argument("--candidate-map", required=True)
    parser.add_argument("--representations", nargs="+", choices=("dreams", "moldeberta", "fingerprint"), default=("dreams", "moldeberta", "fingerprint"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--annotate-peaks",
        action="store_true",
        help="also assign peak subformulas required by FLARE and MVP",
    )
    args = parser.parse_args()

    selected = list(dict.fromkeys(args.representations))
    print(
        f"Precomputing representations for {args.dataset} "
        f"(candidate map: {args.candidate_map})."
    )
    print(f"Selected representations: {', '.join(selected)}")

    total_steps = len(selected) + int(args.annotate_peaks)
    step = 0

    def announce(label):
        nonlocal step
        step += 1
        print(f"\n[{step}/{total_steps}] {label}", flush=True)

    if "dreams" in selected:
        announce("DreaMS spectrum embeddings")
        run_dreams(args.dataset, args.batch_size, args.workers, args.overwrite)
    if "moldeberta" in selected:
        announce("MolDeBERTa molecule embeddings")
        common = dict(
            dataset_name=args.dataset, mol_encoder="moldeberta_base_123m_mtr",
            batch_size=args.batch_size, n_workers=args.workers, overwrite=args.overwrite,
        )
        get_molecule_embeddings(**common)
        get_molecule_embeddings_for_candidates(candidate_map_name=args.candidate_map, **common)
        print("MolDeBERTa preprocessing finished.", flush=True)
    if "fingerprint" in selected:
        announce("Morgan molecular fingerprints")
        common = dict(
            dataset_name=args.dataset, fingerprint="morgan_2_4096",
            batch_size=256, n_workers=args.workers, overwrite=args.overwrite,
        )
        get_molecule_fingerprint(**common)
        get_molecule_fingerprint_for_candidates(candidate_map_name=args.candidate_map, **common)
        print("Morgan fingerprint preprocessing finished.", flush=True)
    if args.annotate_peaks:
        announce("Peak subformula annotations")
        annotate_peaks(
            args.dataset,
            n_threads=args.workers,
            overwrite=args.overwrite,
        )
    print(f"\nFinished preprocessing representations for {args.dataset}.")


if __name__ == "__main__":
    main()
