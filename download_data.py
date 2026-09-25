"""Download the raw MassSpecGym or SpectraVerse dataset."""

import argparse

from preprocessing import download_massspecgym, download_spectraverse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("massspecgym", "spectraverse"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(f"Preparing raw download for {args.dataset}.")
    if args.dataset == "massspecgym":
        download_massspecgym(overwrite=args.overwrite)
    else:
        download_spectraverse(overwrite=args.overwrite)
    print(f"Finished raw download for {args.dataset}.")


if __name__ == "__main__":
    main()
