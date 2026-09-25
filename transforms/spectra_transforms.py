import numpy as np
from preprocessing.definitions import CHEM_ELEMS_MS, CHEM_ELEMS_MS_ABUNDANCE, MAX_MZ, BIN_WIDTH
import re
# remove rdkit warnings (e.g. "Explicit valence for atom # 0 C, is greater than permitted")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class Subformula_Transform:
    '''
    Convert a spectra (list of mz, list of intensities, list of subformulas) to a list of tokens, where each token is a vector of length 2 + number of elements. The first 2 dimensions are the mz and intensity, and the rest are the counts of each element in the subformula.
    '''
    
    def __init__(self, add_mz=True, add_intensity=True):
        if len(CHEM_ELEMS_MS) != len(CHEM_ELEMS_MS_ABUNDANCE):
            raise ValueError(
                "CHEM_ELEMS_MS and CHEM_ELEMS_MS_ABUNDANCE must have the "
                "same length and matching order "
                f"({len(CHEM_ELEMS_MS)} != {len(CHEM_ELEMS_MS_ABUNDANCE)})"
            )
        self.add_mz = add_mz
        self.add_intensity = add_intensity
        
    def get_dim(self):
        return 2 + len(CHEM_ELEMS_MS)
        
    def parse_formula(self, formula: str) -> dict[str, int]:
        """Parse a chemical formula string into element counts."""
        # Match element symbol (capital + optional lowercase) followed by optional count
        pattern = r'([A-Z][a-z]?)(\d*)'
        counts = {elem: 0 for elem in CHEM_ELEMS_MS}
        for match in re.finditer(pattern, formula):
            symbol, count = match.group(1), match.group(2)
            if symbol in counts:
                counts[symbol] += int(count) if count else 1
        return counts

    def __call__(self, mz_list, intensities_list, formulas_list):
        if mz_list is None:
            return np.zeros((1, self.get_dim()), dtype=np.float32)

        n = len(mz_list)
        token_dim = self.get_dim()
        tokens = np.zeros((n, token_dim), dtype=np.float32)

        for i, (mz, intensity, formula) in enumerate(zip(mz_list, intensities_list, formulas_list)):
            tokens[i, 0] = mz
            tokens[i, 1] = intensity

            elem_counts = self.parse_formula(formula)
            for j, elem in enumerate(CHEM_ELEMS_MS):
                tokens[i, 2 + j] = elem_counts[elem]
                
        tokens[:, 0] = tokens[:, 0] / 1000.0 if self.add_mz else 0.0
        tokens[:, 1] = tokens[:, 1] if self.add_intensity else 0.0
        tokens[:, 2:] /= CHEM_ELEMS_MS_ABUNDANCE
        return tokens
    
class BIN_Transform:
    '''
    Transform a spectra to a binned representation. The input is a list of (mz, intensity) pairs, and the output is a vector of length num_bins, where each bin contains the sum of intensities of peaks that fall into that bin. The bins are defined by a specified bin width and maximum mz.
    '''
    
    def __init__(self, max_mz=MAX_MZ, bin_width=BIN_WIDTH):
        self.max_mz = max_mz
        self.bin_width = bin_width
        
    def get_dim(self):
        return int(np.ceil(self.max_mz / self.bin_width))
    
    def __call__(self, ms):
  
        mzs = ms[:, 0]
        intensities = ms[:, 1]
            
        # Calculate the number of bins
        num_bins = int(np.ceil(self.max_mz / self.bin_width))

        # Calculate the bin indices for each mass
        bin_indices = np.floor(mzs / self.bin_width).astype(int)

        # Filter out mzs that exceed max_mz
        
        valid_indices = bin_indices[mzs <= self.max_mz]
        valid_intensities = intensities[mzs <= self.max_mz]

        # Clip bin indices to ensure they are within the valid range
        valid_indices = np.clip(valid_indices, 0, num_bins - 1)

        # Initialize an array to store the binned intensities
        binned_intensities = np.zeros(num_bins)

        # Use np.add.at to sum intensities in the appropriate bins
        np.add.at(binned_intensities, valid_indices, valid_intensities)

        binned_intensities = binned_intensities/np.max(binned_intensities) * 999

        binned_intensities = np.log10(binned_intensities + 1) / 3
        
        return binned_intensities
