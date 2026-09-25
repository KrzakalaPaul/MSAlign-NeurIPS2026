
from rdkit import RDLogger
from tqdm import tqdm
import re
import multiprocessing as mp
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.inchi import MolToInchi
from functools import partial
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from .murcko import murcko_hist
from rdkit.Chem.Descriptors import ExactMolWt
RDLogger.DisableLog('rdApp.*')  # Suppress RDKit warnings

def _smiles_to_mass(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    mol = Chem.AddHs(mol)
    return ExactMolWt(mol)

def _smiles_to_formula(smiles: str):
    """Return formula (with hydrogens) or None if SMILES invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    return rdMolDescriptors.CalcMolFormula(mol)

def _smiles_murcko_hist(smiles: str):
    """Return Murcko histogram or None if SMILES invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return murcko_hist(mol, as_str=True)  # Return as string key for easier splitting

def _smiles_murcko_scaffold(smiles: str):
    """Return canonical Murcko scaffold SMILES or None if SMILES invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)

def _smiles_to_inchi(smi: str) -> str | None:
    """Worker: convert one SMILES string to a standard InChI."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return MolToInchi(mol)      

def normalize_smiles(smi, canonical=True, isomeric=False, sanitize=True, None_For_invalid=False):
    mol = Chem.MolFromSmiles(smi, sanitize=sanitize)
    if mol is not None:
        return Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=isomeric)
    else:
        return smi if not None_For_invalid else None
    
def normalize_smiles_list(smiles_list, 
                          n_threads=16,
                          chunk_size=4096,
                          canonical=True, 
                          isomeric=False, 
                          sanitize=True, 
                          None_For_invalid=False,
                          description="Normalizing SMILES"):
    function = partial(normalize_smiles, canonical=canonical, isomeric=isomeric, sanitize=sanitize, None_For_invalid=None_For_invalid)
    with mp.Pool(processes=n_threads) as pool:
        results = list(tqdm(
            pool.imap(function, smiles_list, chunksize=chunk_size),
            total=len(smiles_list),
            desc=description
        ))
    return results

def normalize_candidate_dict(
    candidate_dict: dict,
    n_threads: int = 16,
    chunk_size: int = 4096,
    canonical: bool = True,
    isomeric: bool = False,
    sanitize: bool = True,
    None_For_invalid: bool = False,
    description: str = "Normalizing candidate map",
) -> dict:
    """
    Normalize a candidate dictionary {smiles: [list of smiles]}.
    Both keys and values (candidate lists) are normalized.
    
    Args:
        candidate_dict (dict): Dictionary mapping a query SMILES to a list of candidate SMILES.
        n_threads (int): Number of parallel workers.
        chunk_size (int): Chunk size for imap.
        canonical (bool): Whether to canonicalize SMILES.
        isomeric (bool): Whether to keep isomeric information.
        sanitize (bool): Whether to sanitize molecules.
        None_For_invalid (bool): Return None for invalid SMILES instead of raising.
        description (str): Description for the progress bar.
    
    Returns:
        dict: Normalized dictionary {normalized_key: [normalized_candidates]}.
    """
    normalize_fn = partial(normalize_smiles, canonical=canonical, isomeric=isomeric, sanitize=sanitize, None_For_invalid=None_For_invalid)

    # Collect all unique SMILES (keys + all values) for a single parallel pass
    keys = list(candidate_dict.keys())
    all_smiles = list({smi for smi in keys + [smi for v in candidate_dict.values() for smi in v]})

    with mp.Pool(processes=n_threads) as pool:
        normalized = list(tqdm(
            pool.imap(normalize_fn, all_smiles, chunksize=chunk_size),
            total=len(all_smiles),
            desc=description,
        ))

    norm_map = dict(zip(all_smiles, normalized))

    print("Rebuilding normalized candidate rows...")
    return {
        norm_map[key]: [norm_map[smi] for smi in candidates]
        for key, candidates in tqdm(
            candidate_dict.items(),
            total=len(candidate_dict),
            desc="Rebuilding candidate map",
            unit="target",
            dynamic_ncols=True,
        )
    }

def compute_formulas(smiles_list, n_threads=16, chunk_size=4096):
    results = []
    with mp.Pool(processes=n_threads) as pool:
        for r in tqdm(
            pool.imap(_smiles_to_formula, smiles_list, chunksize=chunk_size),
            total=len(smiles_list),
            desc="Computing smiles -> formulas"
        ):
            results.append(r)
    return results

def compute_masses(smiles_list, n_threads=16, chunk_size=4096):
    results = []
    with mp.Pool(processes=n_threads) as pool:
        for r in tqdm(
            pool.imap(_smiles_to_mass, smiles_list, chunksize=chunk_size),
            total=len(smiles_list),
            desc="Computing smiles -> masses"
        ):
            results.append(r)
    return results

def compute_inchi(
    smiles_list: list[str],
    n_threads: int = 16,
    chunk_size: int = 4096,
) -> list[str | None]:
    """
    Parallel SMILES → InChI conversion.

    Returns a list of InChI strings (or None for unparseable SMILES),
    preserving input order.
    """
    results = []
    with mp.Pool(processes=n_threads) as pool:
        for r in tqdm(
            pool.imap(_smiles_to_inchi, smiles_list, chunksize=chunk_size),
            total=len(smiles_list),
            desc="Computing SMILES → InChI",
        ):
            results.append(r)
    return results

def compute_murcko_histograms(smiles_list, n_threads=16, chunk_size=4096):
    results = []
    with mp.Pool(processes=n_threads) as pool:
        for r in tqdm(
            pool.imap(_smiles_murcko_hist, smiles_list, chunksize=chunk_size),
            total=len(smiles_list),
            desc="Computing Murcko histograms"
        ):
            results.append(r)
    return results

def compute_murcko_scaffolds(smiles_list, n_threads=16, chunk_size=4096):
    results = []
    with mp.Pool(processes=n_threads) as pool:
        for r in tqdm(
            pool.imap(_smiles_murcko_scaffold, smiles_list, chunksize=chunk_size),
            total=len(smiles_list),
            desc="Computing Murcko scaffolds"
        ):
            results.append(r)
    return results

def process_smiles(smiles, canonical=True, isomeric=False, sanitize=True):
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is not None:
        mol_h = Chem.AddHs(mol)
        formula = rdMolDescriptors.CalcMolFormula(mol_h)
        mass = ExactMolWt(mol_h)
        mass = round(mass, 6)  # Round to 6 decimal places for consistency
        normalized_smi = Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=isomeric)
        return (normalized_smi, formula, mass)
    else:
        return None

def preprocess_smiles_list(smiles_list, n_threads=16, chunk_size=4096,canonical=True,isomeric=False,sanitize=True,description="Preprocessing SMILES",remove_invalid=True):
    
    function = partial(process_smiles, canonical=canonical, isomeric=isomeric, sanitize=sanitize)
        
    results = []
    with mp.Pool(processes=n_threads) as pool:
        for r in tqdm(
            pool.imap(function, smiles_list, chunksize=chunk_size),
            total=len(smiles_list),
            desc=description
        ):
            results.append(r)
    if remove_invalid:
        before_count = len(results)
        results = [r for r in results if r is not None]
        after_count = len(results)
        print(f"Preprocessed {before_count} SMILES, {after_count} valid after normalization.")
    else:
        results = [r if r is not None else (None, None, None) for r in results]  # Replace None with tuple of Nones for consistency
    normalized_smiles, formulas, masses = zip(*results)
    return list(normalized_smiles), list(formulas), list(masses)
        

    
