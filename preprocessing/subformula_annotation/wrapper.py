from .subformula_assign import assign_subforms
from .chem_utils import form_from_smi
import numpy as np


def assign_subformula(smiles, 
                      mass_array,
                      intensity_array,  
                      precursor_mz,
                      adduct, 
                      mass_diff_thresh=15,
                      n_max_peaks=32):
    
    # Normalize intensities
    intensity_array = intensity_array / np.max(intensity_array)
    
    # Concatenate precursor ion
    if precursor_mz is not None:
        mass_array = np.concatenate([mass_array, [precursor_mz]])
        intensity_array = np.concatenate([intensity_array, [1.1]])

    spec = np.stack([mass_array, intensity_array], axis=1)
    formula = form_from_smi(smiles)

    out = assign_subforms(formula, spec, adduct, mass_diff_thresh=mass_diff_thresh)
    mz = np.array(out["mz"])
    intensities = np.array(out["ms2_inten"])
    formulas = np.array(out["formula"])
    # sort by intensity and keep top n peaks
    sort_inds = np.argsort(intensities)[::-1][:n_max_peaks]
    mz = mz[sort_inds]
    intensities = intensities[sort_inds]
    formulas = formulas[sort_inds].tolist()
    return {
        "mz": mz,
        "intensities": intensities,
        "formulas": formulas
    }
