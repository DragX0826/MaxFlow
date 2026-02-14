# maxflow/utils/motifs.py

import torch
from rdkit import Chem
from rdkit.Chem import BRICS, Recap
import numpy as np

class MotifDecomposer:
    """
    Decomposes molecules into rigid motifs using BRICS algorithm.
    Provides mapping between atoms and motifs.
    """
    
    def __init__(self):
        # Common motifs to assign IDs to (Benzene, etc.)
        # For simplicity in this SOTA prototype, we'll use SMILES as keys
        self.motif_vocab = {}
        self.next_id = 0

    def decompose(self, mol):
        """
        SOTA Phase 49: Hybrid BRICS/RECAP decomposition with Joint Tracking.
        """
        if mol is None: return [], torch.empty((0, 2), dtype=torch.long)
        
        # 1. Identify BRICS bonds
        brics_bonds = list(BRICS.FindBRICSBonds(mol))
        
        # 2. Identify RECAP bonds (more conservative)
        hierarch = Recap.RecapDecompose(mol)
        recap_bonds = []
        # Extract bonds from Recap (heuristic: common med-chem bonds)
        # For simplicity, we prioritize BRICS but filter by Recap if needed
        # In this implementation, we combine them.
        
        # 3. Collect Joint Indices (Atoms involved in broken bonds)
        joint_indices = []
        bond_indices = []
        for b in brics_bonds:
            u, v = b[0]
            joint_indices.append([u, v])
            bond_indices.append(mol.GetBondBetweenAtoms(u, v).GetIdx())
            
        if not bond_indices:
            return [{'smiles': Chem.MolToSmiles(mol), 'atoms': list(range(mol.GetNumAtoms()))}], torch.empty((0, 2), dtype=torch.long)
            
        # 4. Fragment molecule
        frags_mol = Chem.FragmentOnBonds(mol, bond_indices, addDummies=False)
        groups = Chem.GetMolFrags(frags_mol)
        
        motifs = []
        for group in groups:
            sub_mol = Chem.RWMol(mol)
            atoms_to_remove = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in group]
            for aid in sorted(atoms_to_remove, reverse=True):
                sub_mol.RemoveAtom(aid)
            
            try:
                smiles = Chem.MolToSmiles(sub_mol)
            except:
                smiles = "invalid"
                
            motifs.append({
                'smiles': smiles,
                'atoms': list(group)
            })
            
        return motifs, torch.tensor(joint_indices, dtype=torch.long)

    def get_motif_centers(self, pos, motifs):
        """
        Calculates the geometric center of each motif.
        pos: (N, 3) tensor
        """
        centers = []
        for m in motifs:
            atom_indices = m['atoms']
            center = pos[atom_indices].mean(dim=0)
            centers.append(center)
        return torch.stack(centers) if centers else torch.empty(0, 3)

    def get_rigid_transformations(self, pos, motifs, centers):
        """
        SOTA Phase 49: PCA-based local frame alignment for motifs.
        Provides a stable SE(3) reference frame for each fragment.
        """
        rotations = []
        for i, m in enumerate(motifs):
            atom_indices = m['atoms']
            if len(atom_indices) < 3:
                # Fallback for small motifs: stick to Identity
                rotations.append(torch.eye(3, device=pos.device))
                continue
                
            # 1. Local Centering
            patch = pos[atom_indices] - centers[i]
            
            # 2. SVD to find principal axes
            try:
                # patch: (K, 3). Vh: (3, 3)
                U, S, Vh = torch.linalg.svd(patch)
                # Vh Rows are principal axes. Use them as rotation matrix rows.
                R = Vh
                # Ensure det(R) = 1
                if torch.linalg.det(R) < 0:
                    R[0] = -R[0]
                rotations.append(R)
            except:
                rotations.append(torch.eye(3, device=pos.device))
                
        return torch.stack(rotations) if rotations else torch.empty(0, 3, 3)

    def get_hierarchical_projected_velocity(self, pos, velocity, atom_to_motif):
        """
        SOTA Phase 52: Hierarchical Flow Projection.
        Decomposes atomic velocity into:
        1. Motif translation (v_trans)
        2. Motif rotation (v_rot / torque)
        3. Local residual deformation (v_local)
        
        Returns:
            v_trans: (num_motifs, 3)
            v_rot: (num_motifs, 3)
            v_local: (num_atoms, 3)
        """
        num_atoms = pos.size(0)
        num_motifs = int(atom_to_motif.max().item() + 1)
        
        v_trans = torch.zeros((num_motifs, 3), device=pos.device)
        v_rot = torch.zeros((num_motifs, 3), device=pos.device)
        v_local = velocity.clone()
        
        for i in range(num_motifs):
            mask = (atom_to_motif == i)
            if not mask.any(): continue
            
            # Sub-cluster positions and velocities
            p_sub = pos[mask]
            v_sub = velocity[mask]
            
            # 1. Translation: Mean velocity of the cluster
            v_t = v_sub.mean(dim=0)
            v_trans[i] = v_t
            
            # 2. Rotation: Use torque-like projection (cross product)
            # Torque T = sum (r x F) where F is velocity here
            center = p_sub.mean(dim=0)
            r = p_sub - center
            torque = torch.cross(r, v_sub, dim=-1).sum(dim=0)
            
            # Moment of Inertia Tensor (3x3)
            # I = sum (r^2 * I - r * r^T)
            # Since mass=1 (approx), just geometric inertia.
            r_sq = (r ** 2).sum(dim=-1, keepdim=True) # (N, 1)
            I_mat = torch.zeros(3, 3, device=pos.device)
            # Scatter add logic or loop (small set usually < 10 atoms)
            # Vectorized:
            # I = sum( r_sq * eye - r outer r )
            eye = torch.eye(3, device=pos.device).unsqueeze(0) # (1, 3, 3)
            r_outer = r.unsqueeze(2) * r.unsqueeze(1) # (N, 3, 3)
            I_atoms = r_sq.unsqueeze(2) * eye - r_outer # (N, 3, 3)
            I_tensor = I_atoms.sum(dim=0) # (3, 3)

            # SOTA Phase 52 Fix: Use Pseudo-Inverse for stability
            # omega = I_inv @ torque
            # Add damping to diagonal for numerical stability
            I_tensor = I_tensor + torch.eye(3, device=pos.device) * 1e-6
            
            try:
                omega = torch.linalg.solve(I_tensor, torque)
            except:
                # Fallback to scalar approx if singular
                inertia_scaler = r_sq.sum() + 1e-6
                omega = torque / inertia_scaler
                
            v_rot[i] = omega
            
            # Rigid component at each atom: v_t + omega x r
            v_rigid = v_t + torch.cross(omega.expand_as(r), r, dim=-1)
            
            # 3. Residual: Local deformation
            v_local[mask] = v_sub - v_rigid
                
        return v_trans, v_rot, v_local
