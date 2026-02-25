import os
import sys
import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Any

from .esm import get_esm_model

logger = logging.getLogger("SAEB-Flow.utils.pdb_io")



def save_points_as_pdb(coords, path, resname="LIG"):
    """Saves a raw coordinate array as a PDB file for external audit (PoseBusters)."""
    with open(path, "w") as f:
        for i, (x, y, z) in enumerate(coords):
            f.write(f"HETATM{i+1:5d}  C   {resname} A   1    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        f.write("END\n")

class RealPDBFeaturizer:
    """
    Downloads, Parses, and Featurizes PDB files.
    Robust to missing residues, alternat locations, and insertions.
    """
    def __init__(self, esm_path="esm_embeddings.pt", config=None, device='cpu'):
        try:
            from Bio.PDB import PDBParser
            self.parser = PDBParser(QUIET=True)
        except ImportError:
            self.parser = None
            logger.error(" PDBParser could not be initialized (BioPython missing).")
        
        self.device = device
        self.aa_map = {
            'ALA':0,'ARG':1,'ASN':2,'ASP':3,'CYS':4,
            'GLN':5,'GLU':6,'GLY':7,'HIS':8,'ILE':9,
            'LEU':10,'LYS':11,'MET':12,'PHE':13,'PRO':14,
            'SER':15,'THR':16,'TRP':17,'TYR':18,'VAL':19,
            # Nucleic Acids (Imp 4 v8.0)
            'A':20, 'U':21, 'G':22, 'C':23, 'T':24, # RNA/DNA single
            'DA':20, 'DU':21, 'DG':22, 'DC':23, 'DT':24, # DNA double
            'ADE':20, 'URA':21, 'GUA':22, 'CYT':23, 'THY':24, # 3-letter
        }
        self.esm_dim = 1280
        self.config = config
        self.esm_embeddings = {}
        if os.path.exists(esm_path):
            try:
                self.esm_embeddings = torch.load(esm_path, map_location=self.device)
                logger.info(f"Loaded {len(self.esm_embeddings)} pre-computed embeddings.")
            except Exception as e:
                logger.warning(f"Failed to load {esm_path}: {e}")

    def _compute_esm_dynamic(self, pdb_id, sequence):
        """ESM Disk Cache: Saves 1-min generation per target."""
        os.makedirs("./cache/esm", exist_ok=True)
        cache_path = f"./cache/esm/{pdb_id}.pt"
        if os.path.exists(cache_path):
            try:
                return torch.load(cache_path, map_location=self.device)
            except Exception as e:
                logger.warning(f" [Cache] Failed to load ESM cache for {pdb_id}: {e}")
            
        if not sequence: return None
        model, alphabet = get_esm_model(device=self.device)
        if model is None: return None
        
        logger.info(f"Dynamically Generating Embeddings for {pdb_id} ({len(sequence)} AA)...")
        try:
            batch_converter = alphabet.get_batch_converter()
            _, _, batch_tokens = batch_converter([(pdb_id, sequence)])
            batch_tokens = batch_tokens.to(self.device)
            
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_representations = results["representations"][33]
            
            feat = token_representations[0, 1 : len(sequence) + 1].float()
            feat = torch.nan_to_num(feat, nan=0.0)
            
            try: torch.save(feat, cache_path)
            except Exception as e:
                logger.warning(f" [Cache] Failed to save ESM cache for {pdb_id}: {e}")
            return feat
        except Exception as e:
            logger.error(f"Dynamic Computation Failed: {e}")
            return None

    def parse(self, pdb_id: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Any]:
        """
        Returns:
            pos_P (M,3), x_P (M,D), q_P (M,), (pocket_center, pos_native), x_L_native (N,D), ligand_template
        """
        path = f"{pdb_id}.pdb"
        kaggle_paths = [
            f"/kaggle/input/pdbbind-v2020/{pdb_id}/{pdb_id}_protein.pdb",
            f"/kaggle/input/pdbbind-v2020/{pdb_id}/{pdb_id}_ligand.pdb",
            f"/kaggle/input/SAEB-Flow-priors/{pdb_id}.pdb",
            path
        ]
        
        found = False
        for kp in kaggle_paths:
            if os.path.exists(kp):
                path = kp
                found = True
                break
        
        if not found:
            try:
                import urllib.request
                logger.info(f" {pdb_id} NOT FOUND locally. Attempting RCSB download...")
                urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", path)
            except Exception as e:
                logger.error(f" CRITICAL DATA FAILURE: {pdb_id} could not be retrieved. internet={sys.flags.ignore_environment == 0}. Falling back to mock data.")
                return self.mock_data()

        try:
            from Bio.PDB.Polypeptide import three_to_one
        except:
            three_to_one = lambda x: 'X'

        try:
            struct = self.parser.get_structure(pdb_id, path)
            coords, feats, charges = [], [], []
            native_ligand = []
            native_elements = [] 
            native_resname = "UNK" # [FIX 9] Initialize to avoid NameError
            res_sequences = []
            atom_to_res_idx = []
            type_map = {'C': 0, 'N': 1, 'O': 2, 'S': 3}

            for model in struct:
                for chain in model:
                    for res in chain:
                        if res.get_resname() in self.aa_map:
                            heavy_atoms = [a for a in res if a.element != 'H']
                            for atom in heavy_atoms:
                                coords.append(atom.get_coord())
                                atom_oh = [0.0] * 4
                                e = atom.element if atom.element in type_map else 'C'
                                atom_oh[type_map.get(e, 0)] = 1.0 # Safer get
                                
                                res_oh = [0.0] * 26 # Expanded for NA (Imp 4 v8.0)
                                res_oh[self.aa_map[res.get_resname().strip()]] = 1.0
                                
                                feats.append(atom_oh + res_oh)
                                atom_to_res_idx.append(len(res_sequences))
                                
                                q = 0.0
                                if res.get_resname() in ['ARG', 'LYS']: q = 1.0 / len(heavy_atoms)
                                elif res.get_resname() in ['ASP', 'GLU']: q = -1.0 / len(heavy_atoms)
                                charges.append(q)
                            
                            try:
                                res_char = three_to_one(res.get_resname())
                            except:
                                res_char = 'X'
                            res_sequences.append(res_char)
                            
                        elif res.get_resname() not in ['HOH', 'WAT', 'NA', 'CL', 'MG', 'ZN', 'SO4', 'PO4', 'K']:
                             # Robust Ligand Detection (Imp 6 v8.0)
                             # Check for non-standard residues that have >5 atoms
                             candidate_atoms = []
                             for atom in res:
                                 if atom.element != 'H':
                                     candidate_atoms.append(atom.get_coord())
                             
                             if len(candidate_atoms) > 4: # Small molecule threshold
                                 # If multiple candidates, pick the largest
                                 if len(candidate_atoms) > len(native_ligand):
                                     native_ligand = candidate_atoms
                                     native_elements = [a.element.strip().upper() for a in res if a.element != 'H']
                                     native_resname = res.get_resname()
                                     logger.info(f"    [Data] Found dominant ligand {native_resname} with {len(native_ligand)} heavy atoms.")
            
            if not native_ligand:
                logger.warning(f"No ligand found in {pdb_id}. Creating mock cloud.")
                native_ligand = np.random.randn(20, 3) + np.mean(coords, axis=0)
            
            esm_feat = None
            if pdb_id in self.esm_embeddings:
                esm_feat = self.esm_embeddings[pdb_id]
            else:
                full_seq = "".join(res_sequences)
                esm_feat = self._compute_esm_dynamic(pdb_id, full_seq)
            
            if esm_feat is not None:
                if esm_feat.size(-1) < 1280:
                    padding = 1280 - esm_feat.size(-1)
                    esm_feat = F.pad(esm_feat, (0, padding))
                elif esm_feat.size(-1) > 1280:
                    esm_feat = esm_feat[..., :1280]
                esm_feat = esm_feat.to(self.device)  # ensure on correct device
                
                atom_esm = [esm_feat[idx] if idx < len(esm_feat) else torch.zeros(1280, device=self.device) for idx in atom_to_res_idx]
                # Move both tensors to device BEFORE concatenation
                atom_oh_t = torch.tensor(np.array(feats), dtype=torch.float32)[:, :4].to(self.device)
                x_P_all = torch.cat([atom_oh_t, torch.stack(atom_esm)], dim=-1)
            else:
                # Handle extended One-Hot padding for missing ESM
                x_full = np.array(feats)
                x_P_all = torch.tensor(x_full, dtype=torch.float32).to(self.device)
                if x_P_all.size(-1) < 1284:
                    x_P_all = F.pad(x_P_all, (0, 1284 - x_P_all.size(-1)))

            if len(native_ligand) > 0:
                pos_native = torch.tensor(np.array(native_ligand), dtype=torch.float32).to(self.device)
                pocket_center = pos_native.mean(0)
            else:
                pos_native = torch.zeros(20, 3).to(self.device)
                pocket_center = torch.tensor(np.array(coords), dtype=torch.float32).mean(0).to(self.device)

            pos_P_all = torch.tensor(np.array(coords), dtype=torch.float32).to(self.device)
            q_P_all = torch.tensor(np.array(charges), dtype=torch.float32).to(self.device)

            x_L_native = torch.zeros(len(native_ligand), 167, device=self.device)
            q_L_native = torch.zeros(len(native_ligand), device=self.device)
            lig_type_map = {'C':0, 'N':1, 'O':2, 'S':3, 'F':4, 'P':5, 'CL':6, 'BR':7, 'I':8}
            for i, el in enumerate(native_elements):
                if el in lig_type_map:
                    x_L_native[i, lig_type_map[el]] = 1.0
                else:
                    x_L_native[i, 0] = 1.0
            
            # [FIX 2] Attempt to extract formal charges if RDKit is available
            try:
                from rdkit import Chem
                mol_l = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False)
                if mol_l:
                    # Very basic formal charge extraction
                    for i, atom in enumerate(mol_l.GetAtoms()):
                        if i < len(q_L_native):
                            q_L_native[i] = float(atom.GetFormalCharge())
            except: pass
            
            ligand_template = None
            try:
                from rdkit import Chem
                mol_full = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False)
                if mol_full:
                    res_mols = Chem.SplitMolByPDBResidues(mol_full)
                    # Priority 1: Match by ResName (Robust for multi-ligand PDBs)
                    for rm in res_mols.values():
                        # Peek first atom to check residue name
                        if rm.GetNumAtoms() > 0:
                            p_info = rm.GetAtomWithIdx(0).GetPDBResidueInfo()
                            if p_info and p_info.GetResidueName().strip() == native_resname.strip():
                                ligand_template = rm
                                logger.info(f"    Successfully matched ligand template by residue name ({native_resname}).")
                                break
                    
                    # Priority 2: Fallback to atom count (Legacy)
                    if ligand_template is None:
                        for rm in res_mols.values():
                            if rm.GetNumAtoms() == len(native_ligand):
                                ligand_template = rm
                                logger.info(f"    Matched ligand template by atom count ({len(native_ligand)}).")
                                break
            except Exception as template_err:
                logger.warning(f"    Template extraction failed: {template_err}.")

            if len(coords) > 1000:
                dist_to_center = np.linalg.norm(np.array(coords) - pocket_center.cpu().numpy(), axis=1)
                shell_idx = np.argsort(dist_to_center)[:1000]
                pos_P = pos_P_all[shell_idx]
                q_P = q_P_all[shell_idx]
                x_P = x_P_all[shell_idx]
            else:
                pos_P = pos_P_all
                q_P = q_P_all
                x_P = x_P_all
            
            use_native_center = (pos_native is not None and self.config and self.config.redocking and self.config.mode == "train")
            if use_native_center and len(pos_native) > 0:
                pocket_center = pos_native.mean(0)
            else:
                pocket_center = pos_P.mean(0) 
            
            pos_P = pos_P - pocket_center
            if pos_native is not None:
                pos_native = pos_native - pocket_center
            
            return pos_P, x_P, q_P, (torch.zeros(3, device=self.device), pos_native), x_L_native, q_L_native, ligand_template
        except Exception as e:
            logger.warning(f" Protein parsing failed: {e}. Falling back to mock data.")
            return self.mock_data()
    
    def mock_data(self):
        P = torch.randn(100, 3).to(self.device)
        # [FIX 10] Correct ESM+OneHot dimension (1284)
        X = torch.randn(100, 1284).to(self.device)
        Q = torch.randn(100).to(self.device)
        C = torch.zeros(3, device=self.device)
        L = torch.randn(25, 3).to(self.device)
        XL = torch.zeros(25, 167).to(device=self.device)
        QL = torch.zeros(25).to(device=self.device)
        return P, X, Q, (C, L), XL, QL, None
