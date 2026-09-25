import numpy as np
from .chem_utils import get_all_subsets, ion_to_mass, clipped_ppm, vec_to_formula

def assign_subforms(form, spec, ion_type, mass_diff_thresh=15):
    """_summary_

    Args:
        form (_type_): _description_
        spec (_type_): _description_
        ion_type (_type_): _description_
        mass_diff_thresh (int, optional): _description_. Defaults to 15.

    Returns:
        _type_: _description_
    """
    # try:
    cross_prod, masses = get_all_subsets(form)
    spec_masses, spec_intens = spec[:, 0], spec[:, 1]

    ion_masses = ion_to_mass[ion_type]
    masses_with_ion = masses + ion_masses
    ion_types = np.array([ion_type] * len(masses_with_ion))

    mass_diffs = np.abs(spec_masses[:, None] - masses_with_ion[None, :])

    formula_inds = mass_diffs.argmin(-1)
    min_mass_diff = mass_diffs[np.arange(len(mass_diffs)), formula_inds]
    rel_mass_diff = clipped_ppm(min_mass_diff, spec_masses)

    # Filter by mass diff threshold (ppm)
    valid_mask = rel_mass_diff < mass_diff_thresh
    spec_masses = spec_masses[valid_mask]
    spec_intens = spec_intens[valid_mask]
    min_mass_diff = min_mass_diff[valid_mask]
    rel_mass_diff = rel_mass_diff[valid_mask]
    formula_inds = formula_inds[valid_mask]

    formulas = np.array([vec_to_formula(j) for j in cross_prod[formula_inds]])
    formula_masses = masses_with_ion[formula_inds]
    ion_types = ion_types[formula_inds]

    # Build mask for uniqueness on formula and ionization
    # note that ionization are all the same for one subformula assignment
    # hence we only need to consider the uniqueness of the formula
    formula_idx_dict = {}
    uniq_mask = []
    for idx, formula in enumerate(formulas):
        uniq_mask.append(formula not in formula_idx_dict)
        gather_ind = formula_idx_dict.get(formula, None)
        if gather_ind is None:
            continue
        spec_intens[gather_ind] += spec_intens[idx]
        formula_idx_dict[formula] = idx

    spec_masses = spec_masses[uniq_mask]
    spec_intens = spec_intens[uniq_mask]
    min_mass_diff = min_mass_diff[uniq_mask]
    rel_mass_diff = rel_mass_diff[uniq_mask]
    formula_masses = formula_masses[uniq_mask]
    formulas = formulas[uniq_mask]
    ion_types = ion_types[uniq_mask]

    # To calculate explained intensity, preserve the original normalized
    # intensity
    if spec_intens.size == 0:
        output_tbl = None
    else:
        output_tbl = {
            "mz": list(spec_masses),
            "ms2_inten": list(spec_intens),
            "mono_mass": list(formula_masses),
            "abs_mass_diff": list(min_mass_diff),
            "mass_diff": list(rel_mass_diff),
            "formula": list(formulas),
            "ions": list(ion_types),
        }
    # except Exception as e:
    #     print(e)
    #     output_tbl = None
    #     print(f"failed to process formula {form}")
    #     pass

    return output_tbl
