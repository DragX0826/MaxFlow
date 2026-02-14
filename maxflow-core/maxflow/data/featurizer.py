# maxflow/data/featurizer.py

import torch
import warnings
from rdkit import Chem
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from torch_geometric.data import Data
from maxflow.utils.constants import amino_acids, ALLOWED_ATOM_TYPES, ATOM_SYM_TO_IDX
from maxflow.utils.chem import get_atom_features
from maxflow.utils.motifs import MotifDecomposer

# Suppress Biopython warnings
warnings.simplefilter('ignore', PDBConstructionWarning)

class FlowData(Data):
    """
    Custom PyG Data class for MaxFlow.
    Ensures that custom attributes like edge_index_L are correctly
    incremented during collation (Batch.from_data_list).
    """
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_L':
            return self.num_nodes_L
        return super().__inc__(key, value, *args, **kwargs)

class ProteinLigandFeaturizer:
    def __init__(self):
        self.parser = PDBParser(QUIET=True)
        self.aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
        self.decomposer = MotifDecomposer()
        # [SOTA Fix] Coarse-Grained Electrostatics (Residue Charges)
        self.charge_map = {'ARG': 1.0, 'LYS': 1.0, 'ASP': -1.0, 'GLU': -1.0, 'HIS': 0.5} # pH 7.4 approx

    def get_residue_features(self, structure):
        """
        Extracts C-alpha coordinates and Amino Acid one-hot features.
        """
        pos_list = []
        feat_list = []
        charge_list = []
        metal_pos_list = []
        
        metals = {'ZN', 'FE', 'MN', 'CU', 'MG', 'CA'}
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Skip heteroatoms (water, ligands)
                    if residue.id[0] != ' ':
                        continue
                    
                    # Try to get C-alpha
                    if 'CA' in residue:
                        pos = residue['CA'].get_coord()
                        pos_list.append(pos)
                        
                        # One-hot encoding
                        resname = residue.get_resname()
                        idx = self.aa_to_idx.get(resname, len(amino_acids)) # Unknown maps to last + 1? Or 0? 
                        # Let's map unknown to specific or ignore. 
                        # For simplicity, if unknown, use a zero vector or extra class. 
                        # Let's use an extra class for unknown.
                        # Actually constants has 20 AAs. Let's make encoding size 21.
                        
                        one_hot = [0.0] * (len(amino_acids) + 1)
                        if idx < len(amino_acids):
                            one_hot[idx] = 1.0
                        else:
                            one_hot[-1] = 1.0 # Unknown
                        
                        feat_list.append(one_hot)
                        
                        # [SOTA Fix] Charge
                        charge = self.charge_map.get(resname, 0.0)
                        charge_list.append(charge)
                    
                    # SOTA Phase 32: Detect Metals (usually HETATMs)
                    elif residue.get_resname().strip().upper() in metals:
                         for atom in residue:
                              metal_pos_list.append(atom.get_coord())
        
        if len(pos_list) == 0:
            return None, None, None

        return (torch.tensor(feat_list, dtype=torch.float32), 
                torch.tensor(pos_list, dtype=torch.float32),
                torch.tensor(charge_list, dtype=torch.float32),
                torch.tensor(metal_pos_list, dtype=torch.float32) if metal_pos_list else None)

    def estimate_surface_normals(self, pos):
        """
        SOTA Phase 48: Fast Covariance-based Normal Estimation.
        Moved from Runtime to Pre-computation to eliminate SVD overhead.
        """
        if pos.size(0) == 0: return torch.zeros_like(pos)
        
        dist = torch.cdist(pos, pos)
        mask = (dist < 6.5) & (dist > 0)
        
        normals = []
        global_center = pos.mean(dim=0)
        
        for i in range(pos.size(0)):
            neighbors = pos[mask[i]]
            if neighbors.size(0) >= 3:
                # 1. Local Centering
                patch = neighbors - neighbors.mean(dim=0)
                # 2. Covariance Matrix C = X^T * X (3x3)
                # Faster than full SVD for small K
                cov = torch.matmul(patch.T, patch)
                try:
                    # Eigen decomposition of 3x3 matrix is very fast
                    L, V = torch.linalg.eigh(cov)
                    n = V[:, 0] # Eigenvector with smallest eigenvalue
                    
                    if torch.dot(n, pos[i] - global_center) < 0:
                        n = -n
                    normals.append(n)
                except:
                    n = pos[i] - global_center
                    normals.append(n / (torch.norm(n) + 1e-6))
            else:
                n = pos[i] - global_center
                normals.append(n / (torch.norm(n) + 1e-6))
        
        return torch.stack(normals)

    def is_organic_molecule(self, mol):
        """Checks if all atoms in the molecule are from the organic vocabulary."""
        if mol is None: return False
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in ALLOWED_ATOM_TYPES:
                return False
        return True

    def __call__(self, pdb_path: str, sdf_path: str):
        # 1. Load Ligand
        mol = Chem.MolFromMolFile(sdf_path)
        if not self.is_organic_molecule(mol):
            return None # Skip inorganic clusters or failed parse
        
        x_L, pos_L, edge_index_L = get_atom_features(mol)
        
        # Add categorical target for training classification head
        atom_types = torch.tensor([ATOM_SYM_TO_IDX[a.GetAtomicNum()] for a in mol.GetAtoms()], dtype=torch.long)
        
        # 2. Load Protein
        try:
            structure = self.parser.get_structure('protein', pdb_path)
        except Exception:
            return None
            
        x_P, pos_P, q_P, pos_metals = self.get_residue_features(structure)
        
        if x_P is None:
            return None

        # 3. Motif Decomposition (Phase 30/49: Hybrid + Joint Tracking)
        motifs, joint_indices = self.decomposer.decompose(mol)
        atom_to_motif = torch.zeros(mol.GetNumAtoms(), dtype=torch.long)
        for m_idx, m in enumerate(motifs):
            for a_idx in m['atoms']:
                atom_to_motif[a_idx] = m_idx
                
        # 5. Pre-calculate Surface Normals for Protein (Optimization P48)
        normals_P = self.estimate_surface_normals(pos_P)
        
        # 6. Create PyG Data
        # Phase 63: Use FlowData for correct edge_index_L collation
        data = FlowData(
            x_L=x_L,
            pos_L=pos_L,
            atom_types=atom_types, # Target labels
            edge_index_L=edge_index_L,
            x_P=x_P,
            pos_P=pos_P,
            q_P=q_P, # [SOTA Fix] Explicit Charges
            normals_P=normals_P, # Pre-computed!
            pocket_center=pos_L.mean(dim=0, keepdim=True),
            atom_to_motif=atom_to_motif,
            joint_indices=joint_indices, # SOTA Phase 49
            num_motifs=torch.tensor([len(motifs)], dtype=torch.long),
            pos_metals=pos_metals,
            num_nodes_L=torch.tensor([x_L.size(0)], dtype=torch.long),
            num_nodes_P=torch.tensor([x_P.size(0)], dtype=torch.long)
        )
        
        return data
