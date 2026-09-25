"""Lazy public preprocessing API.

Keeping imports inside these wrappers is intentional: training from released
representations must not import Transformers or its dependencies. DreaMS is
even more strongly isolated in ``encode_dreams.py``.
"""


def download_massspecgym(*args, **kwargs):
    from .download import download_massspecgym as function
    return function(*args, **kwargs)


def download_spectraverse(*args, **kwargs):
    from .download import download_spectraverse as function
    return function(*args, **kwargs)


def download_massspecgym_official_candidate_map(*args, **kwargs):
    from .download import download_massspecgym_official_candidate_map as function
    return function(*args, **kwargs)


def process_smiles(*args, **kwargs):
    from .process_smiles import process_smiles as function
    return function(*args, **kwargs)


def split(*args, **kwargs):
    from .split import split as function
    return function(*args, **kwargs)


def prepare_candidates(*args, **kwargs):
    from .prepare_candidates import prepare_candidates as function
    return function(*args, **kwargs)


def annotate_peaks(*args, **kwargs):
    from .annotate_peaks import annotate_peaks as function
    return function(*args, **kwargs)


def get_molecule_embeddings(*args, **kwargs):
    from .encode_mol import get_molecule_embeddings as function
    return function(*args, **kwargs)


def get_molecule_embeddings_for_candidates(*args, **kwargs):
    from .encode_mol import get_molecule_embeddings_for_candidates as function
    return function(*args, **kwargs)


def get_molecule_fingerprint(*args, **kwargs):
    from .fingerprint_mol import get_molecule_fingerprint as function
    return function(*args, **kwargs)


def get_molecule_fingerprint_for_candidates(*args, **kwargs):
    from .fingerprint_mol import get_molecule_fingerprint_for_candidates as function
    return function(*args, **kwargs)
