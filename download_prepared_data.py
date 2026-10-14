"""Download the processed MSAlign datasets released on Zenodo."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from tqdm import tqdm


ARCHIVES = {
    "massspecgym": {
        "url": "https://zenodo.org/api/records/22830464/files/massspecgym.zip/content",
        "size": 1_078_913_065,
        "md5": "96e8b1885cf4120aaef93154ce809cbd",
    },
    "spectraverse": {
        "url": "https://zenodo.org/api/records/22830464/files/spectraverse.zip/content",
        "size": 2_237_587_937,
        "md5": "729425411e4e5da92902782918116dbb",
    },
}


def _valid_dataset(path: Path, dataset: str) -> bool:
    common = ("metadata.csv", "unique_smiles.csv", "spectra.npy")
    if not all((path / name).is_file() for name in common):
        return False
    if dataset == "massspecgym":
        required = (
            "splits/formula_seed0.csv",
            "splits/mces_1.csv",
            "candidates/official_candidates_by_mass/map.json",
        )
    else:
        required = (
            "splits/formula_seed0.csv",
            "candidates/256_candidates_by_mass/map.json",
        )
    return all((path / name).is_file() for name in required)


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - verifies the published archive checksum
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "MSAlign/1.0"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as output:
        total = int(response.headers.get("Content-Length", expected_size))
        with tqdm(
            total=total,
            desc=f"Downloading {destination.name}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
        ) as progress:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                progress.update(len(chunk))
    partial.replace(destination)


def _safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            path = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(mode):
                raise ValueError(f"Unsafe ZIP member: {member.filename!r}")
        for member in tqdm(
            handle.infolist(),
            desc=f"Extracting {archive.name}",
            unit="file",
            dynamic_ncols=True,
        ):
            handle.extract(member, destination)


def download_prepared_dataset(
    dataset: str,
    *,
    data_root: Path = Path("data"),
    archive_root: Path = Path("downloads"),
    keep_archive: bool = False,
) -> Path:
    """Download, verify, and safely install one processed dataset."""

    target = data_root / dataset
    if _valid_dataset(target, dataset):
        print(f"Prepared dataset already exists; skipping download: {target}")
        return target
    if target.exists():
        raise RuntimeError(
            f"Incomplete dataset directory already exists: {target}. "
            "Move it aside before installing the released archive."
        )

    definition = ARCHIVES[dataset]
    archive = archive_root / f"{dataset}.zip"
    if not archive.is_file():
        _download(definition["url"], archive, definition["size"])
    print(f"Verifying {archive}...")
    if archive.stat().st_size != definition["size"]:
        raise ValueError(f"Unexpected archive size: {archive}")
    if _md5(archive) != definition["md5"]:
        raise ValueError(f"Archive checksum mismatch: {archive}")

    data_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{dataset}-", dir=data_root) as temporary:
        temporary = Path(temporary)
        _safe_extract(archive, temporary)
        choices = (
            temporary / dataset,
            temporary / "data" / dataset,
            temporary,
        )
        source = next((path for path in choices if _valid_dataset(path, dataset)), None)
        if source is None:
            raise ValueError(
                f"{archive} does not contain the expected {dataset} dataset layout"
            )
        if source == temporary:
            target.mkdir()
            for child in list(temporary.iterdir()):
                shutil.move(str(child), str(target / child.name))
        else:
            shutil.move(str(source), str(target))

    if not _valid_dataset(target, dataset):
        raise RuntimeError(f"Installed dataset failed validation: {target}")
    if not keep_archive:
        archive.unlink()
    print(f"Prepared dataset is ready: {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=tuple(ARCHIVES))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--archive-root", type=Path, default=Path("downloads"))
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()
    download_prepared_dataset(
        args.dataset,
        data_root=args.data_root,
        archive_root=args.archive_root,
        keep_archive=args.keep_archive,
    )


if __name__ == "__main__":
    main()
