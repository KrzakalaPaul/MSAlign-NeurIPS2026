import numpy as np
from preprocessing.definitions import CHEM_ELEMS_MOL
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator
# remove rdkit warnings (e.g. "Explicit valence for atom # 0 C, is greater than permitted")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class MorganFingerprintTransform():
    '''
    Convert a SMILES string to a Morgan fingerprint vector of specified size and radius.
    '''
    
    def __init__(self, fp_size=4096, radius=2):
        self.fp_size = fp_size
        self.radius = radius

    def get_dim(self):
        return self.fp_size

    def __call__(self, smiles):
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                return np.zeros(self.fp_size, dtype=np.int32)

            generator = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.fp_size)
            fp = generator.GetFingerprint(mol)

            fp_np = np.zeros(self.fp_size, dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_np)
            fp = fp_np
            
            return fp

        except Exception as e:
            return np.zeros(self.fp_size, dtype=np.int32)

class MoleculeToGraph():
    '''
    Convert a SMILES string to a DGL graph with node and edge features.
    '''
    def __init__ (self):
        # DGL is optional for MSAlign and EmbCos; import it only for graph baselines.
        import dgllife.utils as chemutils
        self.chemutils = chemutils
        self.atom_feature = "full"
        self.bond_feature = "full"
        self.node_featurizer = self._get_atom_featurizer(element_list=CHEM_ELEMS_MOL)
        self.edge_featurizer = self._get_bond_featurizer()
        
    def get_dim(self):
        return 78
    
    def __call__(self, smiles:str):
        chemutils = self.chemutils
        mol = Chem.MolFromSmiles(smiles)
        graph = chemutils.mol_to_bigraph(mol, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer, add_self_loop=True,num_virtual_nodes = 0, canonical_atom_order=False)
        return graph

    def _get_atom_featurizer(self, element_list) -> dict:
        chemutils = self.chemutils
        feature_mode = self.atom_feature
        atom_mass_fun = chemutils.ConcatFeaturizer(
            [chemutils.atom_mass]
        )
        def atom_bond_type_one_hot(atom):
            bs = atom.GetBonds()
            if not bs:
                return [False] * 4
            bt = np.array([chemutils.bond_type_one_hot(b) for b in bs])
            return [any(bt[:, i]) for i in range(bt.shape[1])]

        def atom_type_one_hot(atom):
            return chemutils.atom_type_one_hot(
                # dgllife appends an unknown sentinel when encode_unknown=True;
                # pass a copy so it cannot mutate the global element schema.
                atom, allowable_set=list(element_list), encode_unknown=True
            )
        
        if feature_mode == 'light':
            atom_featurizer_funs = chemutils.ConcatFeaturizer([
                chemutils.atom_mass,
                atom_type_one_hot
            ])
        elif feature_mode == 'full':
            atom_featurizer_funs = chemutils.ConcatFeaturizer([
                chemutils.atom_mass,
                atom_type_one_hot, 
                atom_bond_type_one_hot,
                chemutils.atom_degree_one_hot, 
                chemutils.atom_total_degree_one_hot,
                chemutils.atom_explicit_valence_one_hot,
                chemutils.atom_implicit_valence_one_hot,
                chemutils.atom_hybridization_one_hot,
                chemutils.atom_total_num_H_one_hot,
                chemutils.atom_formal_charge_one_hot,
                chemutils.atom_num_radical_electrons_one_hot,
                chemutils.atom_is_aromatic_one_hot,
                chemutils.atom_is_in_ring_one_hot,
                chemutils.atom_chiral_tag_one_hot
            ])
        elif feature_mode == 'medium':
            atom_featurizer_funs = chemutils.ConcatFeaturizer([
                chemutils.atom_mass,
                atom_type_one_hot, 
                atom_bond_type_one_hot,
                chemutils.atom_total_degree_one_hot,
                chemutils.atom_total_num_H_one_hot,
                chemutils.atom_is_aromatic_one_hot,
                chemutils.atom_is_in_ring_one_hot,
            ])
        return chemutils.BaseAtomFeaturizer(
        {"h": atom_featurizer_funs, 
        "m": atom_mass_fun}
    )

    def _get_bond_featurizer(self, self_loop=True) -> dict:
        chemutils = self.chemutils
        feature_mode = self.bond_feature
        if feature_mode == 'light':
            return chemutils.BaseBondFeaturizer(
                featurizer_funcs = {'e': chemutils.ConcatFeaturizer([
                    chemutils.bond_type_one_hot
                ])}, self_loop = self_loop
            )
        elif feature_mode == 'full':
            return chemutils.CanonicalBondFeaturizer(
                bond_data_field='e', self_loop = self_loop
            )
