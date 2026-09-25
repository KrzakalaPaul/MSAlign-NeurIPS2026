from tqdm import tqdm
import multiprocessing as mp
from preprocessing.subformula_annotation.wrapper import assign_subformula
from preprocessing.definitions import N_HIGHEST_PEAKS_FOR_ANNOTATION, MASS_DIFF_THRESH_FOR_ANNOTATION
import json
import numpy as np
import pandas as pd
from rdkit import RDLogger
import os
RDLogger.DisableLog('rdApp.*')  # Suppress RDKit warnings


def _annotate_single(args):
    """Helper for multiprocessing — must be top-level for pickling."""
    i, smiles_i, mass_array, intensity_array, precursor_mz_i, adduct_i, mass_diff_thresh, n_max_peaks = args
    try:
        annotation = assign_subformula(
            smiles_i,
            mass_array,
            intensity_array,
            precursor_mz_i,
            adduct_i,
            mass_diff_thresh=mass_diff_thresh,
            n_max_peaks=n_max_peaks
        )
    except Exception as e:
        print(f"Error at index {i}: {e}")
        annotation = None
    return annotation

def annotate_peaks(dataset_name, n_threads=None, chunksize=256, overwrite=False):
    
    if os.path.exists(f"data/{dataset_name}/annotated_peaks.json") and not overwrite:
        print(f"Annotated peaks already exist for {dataset_name}. Skipping annotation.")
        return
    
    spectra = np.load(f"data/{dataset_name}/spectra.npy")  # [N, n_peaks, 2]
    meta = pd.read_csv(f"data/{dataset_name}/metadata.csv")
    smiles = meta['smiles'].tolist()
    precursor_mz = meta['precursor_mz'].tolist()
    adduct = meta['adduct'].tolist()

    print(
        f"Annotating {len(spectra):,} spectra for {dataset_name} with "
        f"{n_threads or mp.cpu_count()} CPU worker(s)."
    )

    args_list = [
        (i, smiles[i], spectra[i, :, 0], spectra[i, :, 1],
        precursor_mz[i], adduct[i], MASS_DIFF_THRESH_FOR_ANNOTATION, N_HIGHEST_PEAKS_FOR_ANNOTATION)
        for i in range(len(spectra))
    ]

    with mp.Pool(processes=n_threads) as pool:
        all_annotations = list(tqdm(
            pool.imap(_annotate_single, args_list, chunksize=chunksize),
            total=len(spectra),
            desc=f"Annotating spectra {dataset_name}",
            unit="spectrum",
            dynamic_ncols=True,
        ))

    all_mz = [ann['mz'].tolist() if ann is not None else None for ann in all_annotations]
    all_intensities = [ann['intensities'].tolist() if ann is not None else None for ann in all_annotations]
    all_formulas = [ann['formulas'] if ann is not None else None for ann in all_annotations]
    count_none = sum(1 for ann in all_annotations if ann is None)
    print(f"Assigned subformulas to {len(all_annotations) - count_none} out of {len(all_annotations)} spectra. "
        f"{count_none} spectra had no assigned subformula.")
    
    data = {
        'mz':          all_mz,
        'intensities': all_intensities,
        'subformulas': all_formulas,
    }
    
    output_path = f"data/{dataset_name}/annotated_peaks.json"
    with open(output_path, 'w') as f:
        json.dump(data, f)
    print(f"Wrote peak annotations to {output_path}")
