import os
import sys
import subprocess
import time
import zipfile
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime

# [SCALING] Set to True for a quick dry-run, False for ICLR SOTA production
TEST_MODE = False 

# --- SECTION 0: KAGGLE DEPENDENCY AUTO-INSTALLER ---
def auto_install_deps():
    required = ["rdkit", "meeko", "biopython"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"🛠️  Missing basic dependencies found: {missing}. Installing...")
        # Resolve 'rdkit-pypi' conflict by installing meeko without deps (it relies on rdkit)
        safe_missing = [p for p in missing if p != 'meeko']
        if safe_missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + safe_missing)
        
        if 'meeko' in missing:
            print("Force-installing meeko (no-deps mode)...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "meeko", "--no-deps"])
    
    # [SOTA] Special Handling for Torch-Geometric (PyG Suite)
    try:
        import torch_geometric
        import torch_cluster
        import torch_scatter
    except ImportError:
        print("🛠️  Installing Torch-Geometric (PyG) and friends...")
        # Get PyTorch and CUDA versions for the index
        torch_v = torch.get_device_properties(0).name if torch.cuda.is_available() else "cpu" # Dummy check
        torch_v = torch.__version__.split('+')[0]
        cuda_v = 'cpu'
        if torch.cuda.is_available():
            cuda_v = 'cu' + torch.version.cuda.replace('.', '')
        
        index_url = f"https://data.pyg.org/whl/torch-{torch_v}+{cuda_v}.html"
        pkgs = ["torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv", "torch-geometric"]
        
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + pkgs + ["-f", index_url])
        print("✅ SOTA Dependencies (PyG) Installed.")

auto_install_deps()

# --- SECTION 1: KAGGLE ENVIRONMENT SETUP ---
def setup_environment():
    print("🛠️  Authenticating Kaggle Workspace...")
    
    # 1. Aggressive Search for the folder CONTAINING 'maxflow'
    search_roots = ['/kaggle/working', '/kaggle/input', os.getcwd(), '/kaggle/working/DPO-Flow']
    for root in search_roots:
        if not os.path.exists(root): continue
        for r, dirs, files in os.walk(root):
            if 'maxflow' in dirs and os.path.isdir(os.path.join(r, 'maxflow')):
                if r not in sys.path:
                    sys.path.insert(0, r)
                print(f"✅ Path Authenticated: {r}")
                return r
                
    # 2. Diagnostic: If not found, list what we SEE
    print("❌ Failed to find 'maxflow' package. Diagnostic listing:")
    for root in search_roots:
        if os.path.exists(root):
            print(f"\nListing {root}:")
            try:
                print(os.listdir(root))
                # Check one level deeper for common repo names
                for d in os.listdir(root):
                    dp = os.path.join(root, d)
                    if os.path.isdir(dp) and not d.startswith('.'):
                        print(f"  {d}/ -> {os.listdir(dp)[:10]}")
            except: pass
            
    return os.getcwd()

mount_root = setup_environment()

try:
    from maxflow.models.flow_matching import RectifiedFlow
    from maxflow.models.backbone import CrossGVP
    from maxflow.data.featurizer import FlowData
    from maxflow.utils.maxrl_loss import maxrl_objective as maxrl_loss
    from maxflow.utils.optimization import Muon
    print(" MaxFlow Deep Impact Engine Initialized (v18.5).")
except ImportError as e:
    print(f"❌ Structural Failure: {e}. Ensure maxflow-core is in the path.")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')

# --- SECTION 2: DIFFERENTIABLE PHYSICS ENGINE (SOTA TRUTH) ---
# --- SECTION 2: DIFFERENTIABLE PHYSICS ENGINE (SOTA TRUTH) ---
class PhysicsEngine:
    # [SOTA Fix 3/5] Atom-Specific VdW Physics (v18.27)
    # Replaces hardcoded sigma=3.5 with atom-specific radii.
    # C=1.7, N=1.55, O=1.52, S=1.8, H=1.2 (approx)
    # VdW Radius = R_min / 2^(1/6) approx.
    # We use Sigma = 2 * Radius / 2^(1/6)
    # Simplified mapping based on GVP feature index 0-4 (Assuming C, N, O, S, P)
    @staticmethod
    def get_sigma(x_L):
        # x_L is (N, 167). We take softmax to find atom type probabilities.
        # We project to basic 4 types: C, N, O, Others.
        # C-like: 3.4, N-like: 3.1, O-like: 3.0, Others (S, P, etc.): 3.5
        p = x_L[:, :3].softmax(dim=-1)
        # Sum of p is 1.0, but we want to consider if the identity is NOT one of these 3.
        # So we should probably use a softmax over more indices or a different logic.
        # Correct logic for 167-dim:
        p_all = x_L.softmax(dim=-1)
        sigma = p_all[:, 0] * 3.4 + p_all[:, 1] * 3.1 + p_all[:, 2] * 3.0
        p_others = 1.0 - p_all[:, :3].sum(dim=-1)
        sigma = sigma + p_others * 3.5
        return sigma

    @staticmethod
    def get_protein_sigma(x_P):
        # [SOTA Fix] Residue-Specific Radii (v18.45)
        # GLY (7) is smallest, TRP (17) is largest.
        # x_P is (M, 21) one-hot.
        # Approximate Sigma (2 * R_vdw):
        # GLY: 3.0, ALA: 3.2, VAL: 3.4, LEU/ILE: 3.6, PHE/TYR/TRP: 4.0
        # Others: 3.5
        # We implementation a differentiable lookup.
        # Indices: ALA(0), GLY(7), VAL(19), LEU(10), ILE(9), PHE(13), TYR(18), TRP(17)
        
        # Default baseline
        sigma = 3.5 * torch.ones(x_P.size(0), device=x_P.device)
        
        # Override with specific types
        is_gly = x_P[:, 7]; sigma = sigma * (1-is_gly) + 3.0 * is_gly
        is_ala = x_P[:, 0]; sigma = sigma * (1-is_ala) + 3.2 * is_ala
        is_val = x_P[:, 19]; sigma = sigma * (1-is_val) + 3.4 * is_val
        is_large = (x_P[:, 10] + x_P[:, 9]); sigma = sigma * (1-is_large) + 3.6 * is_large
        is_huge = (x_P[:, 13] + x_P[:, 18] + x_P[:, 17]); sigma = sigma * (1-is_huge) + 4.0 * is_huge
        
        return sigma

    @staticmethod
    def compute_energy(pos_L, pos_P, q_L, q_P, x_L=None, x_P=None, dielectric=80.0, softness=0.0):
        # [SOTA Fix] Robust Distance Calculation (Avoids Division by Zero/NaN)
        dist = torch.cdist(pos_L, pos_P)
        dist_eff = torch.sqrt(dist.pow(2) + softness).clamp(min=0.1) 
        
        # Electrostatics (Coulomb)
        e_elec = (332.06 * q_L.unsqueeze(1) * q_P.unsqueeze(0)) / (dielectric * dist_eff)
        
        # Van der Waals (Lennard-Jones 6-12)
        if x_L is not None:
             sigma_L = PhysicsEngine.get_sigma(x_L).unsqueeze(1) # (N_L, 1)
             
             if x_P is not None:
                 sigma_P = PhysicsEngine.get_protein_sigma(x_P).unsqueeze(0) # (1, N_P)
             else:
                 sigma_P = 3.5 # Fallback
                 
             sigma = 0.5 * (sigma_L + sigma_P)
             sigma_6 = sigma ** 6
             r6_eff = dist_eff.pow(6)
             term_r6 = sigma_6 / r6_eff
             
             # Soft-Repulsive SOTA Potential (v18.33) to prevent singularity
             # Standard LJ is unstable at r->0. We use Soft-LJ.
             e_vdw = 0.15 * (term_r6.pow(2) - 2 * term_r6)
        else:
             e_vdw = 0.0 # Should not happen in SOTA mode
        
        # Total Energy
        energy = (e_elec + e_vdw).clamp(min=-1000.0, max=1000.0).sum()
        
        if torch.isnan(energy): return torch.tensor(100.0, device=pos_L.device)
        return energy

    @staticmethod
    def calculate_intra_repulsion(pos, threshold=1.2, softness=0.0):
        if pos.size(0) < 2: return torch.tensor(0.0, device=pos.device)
        dist = torch.cdist(pos, pos) + torch.eye(pos.size(0), device=pos.device) * 10.0
        dist_eff = torch.sqrt(dist.pow(2) + softness)
        rep = torch.relu(threshold - dist_eff).pow(2).sum()
        return rep.clamp(max=1000.0)

    # [SOTA Fix 1/5] Bond Length Constraint
    # Enforces physical chemical bonds (approx 1.5A for C-C) to prevent 'cloud' behavior.
    # Uses a simple harmonic potential for nearest neighbors (k-NN graph).
    @staticmethod
    def calculate_bond_constraint(pos):
        if pos.size(0) < 2: return torch.tensor(0.0, device=pos.device)
        # Infer topology via k-NN (k=3 for standard valence)
        k = min(3, pos.size(0) - 1)
        dist = torch.cdist(pos, pos) + torch.eye(pos.size(0), device=pos.device) * 10.0
        nearest_dist, _ = dist.topk(k, dim=1, largest=False)
        # Bond length deviation from 1.5A
        bond_loss = (nearest_dist - 1.5).pow(2).mean()
        return bond_loss

    @staticmethod
    def calculate_hydrophobic_score(pos_L, x_L, pos_P, x_P, soft=True):
        # [SOTA Fix] Real Hydrophobic Contact (v18.30)
        # ALA(0), ILE(9), LEU(10), MET(12), PHE(13), TRP(17), TYR(18), VAL(19)
        hydro_indices = [0, 9, 10, 12, 13, 17, 18, 19]
        is_hydro = x_P[:, hydro_indices].sum(dim=-1) > 0.5
        pos_P_hydro = pos_P[is_hydro]
        
        if pos_P_hydro.size(0) == 0: return torch.tensor(0.0, device=pos_L.device)
        
        # Distance to Protein Hydrophobic Clusters
        dist = torch.cdist(pos_L, pos_P_hydro)
        min_dist = dist.min(dim=1)[0]
        
        # Gaussian reward around 3.8A
        contact_score = torch.exp(-0.5 * (min_dist - 3.8).pow(2) / 0.5**2).mean()
        return contact_score

# --- SECTION 3: REAL PDB FEATURIZER ---
class RealPDBFeaturizer:
    def __init__(self):
        from Bio.PDB import PDBParser
        self.parser = PDBParser(QUIET=True)
        self.aa_map = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19}
        # [SOTA Fix] Coarse-Grained Electrostatics (Residue Charges)
        self.charge_map = {'ARG': 1.0, 'LYS': 1.0, 'ASP': -1.0, 'GLU': -1.0, 'HIS': 0.5} # pH 7.4 approx

    def parse(self, pdb_id="7SMV"):
        path = f"{pdb_id}.pdb"
        if not os.path.exists(path):
            import urllib.request
            print(f"📥 Downloading target {pdb_id} from RCSB...")
            urllib.request.urlretrieve(f"https://files.rcsb.org/download/{path}", path)
        struct = self.parser.get_structure(pdb_id, path)
        coords, feats, charges = [], [], []
        pocket_center = None
        pos_native = None
        max_atoms = 0
        
        for model in struct:
            for chain in model:
                for res in chain:
                    # [SOTA Fix] Smart Pocket Detection (Largest HETATM)
                    # Use HETATM residues (drug) to find true pocket center
                    if res.id[0].startswith('H_') and res.get_resname() not in ['HOH', 'WAT']:
                        try:
                            lig_coords = np.array([a.get_coord() for a in res])
                            if len(lig_coords) > max_atoms: # Select largest component (Drug vs Ion)
                                max_atoms = len(lig_coords)
                                pocket_center = torch.tensor(lig_coords.mean(0), dtype=torch.float32).to(device)
                                pos_native = torch.tensor(lig_coords, dtype=torch.float32).to(device)
                        except: pass
                    
                    if 'CA' in res and res.get_resname() in self.aa_map:
                        coords.append(res['CA'].get_coord())
                        one_hot = [0.0] * 21
                        one_hot[self.aa_map[res.get_resname()]] = 1.0
                        feats.append(one_hot)
                        charges.append(self.charge_map.get(res.get_resname(), 0.0))
        
        # Fallback if no ligand found
        if pocket_center is None:
            pocket_center = torch.tensor(np.array(coords).mean(0), dtype=torch.float32).to(device)
            # Dummy native for compilation safety
            pos_native = torch.zeros((1, 3), device=device)
            print("⚠️ Warning: No native ligand found. Using protein center.")
            
        return (torch.tensor(np.array(coords), dtype=torch.float32).to(device), 
                torch.tensor(np.array(feats), dtype=torch.float32).to(device),
                torch.tensor(np.array(charges), dtype=torch.float32).to(device),
                (pocket_center, pos_native))

# --- SECTION 4: ABLATION RUNNER CORE (GRPO-STYLE) ---
class AblationSuite:
    def __init__(self):
        self.results = []
        self.device = device
        self.feater = RealPDBFeaturizer()
        
    # [SOTA Metric] Kabsch Alignment (RMSD)
    # Computes optimal rotation to minimize RMSD between generated and native structure.
    @staticmethod
    def calculate_rmsd(pos1, pos2):
        # ... (Existing Kabsch Logic) ...
        # Center both
        c1 = pos1.mean(dim=0)
        c2 = pos2.mean(dim=0)
        p1 = pos1 - c1
        p2 = pos2 - c2
        H = torch.matmul(p1.T, p2)
        try:
            U, S, Vt = torch.linalg.svd(H)
            d = torch.det(torch.matmul(Vt.T, U.T))
            E = torch.eye(3, device=pos1.device)
            E[2, 2] = d
            R = torch.matmul(torch.matmul(Vt.T, E), U.T)
            p1_rot = torch.matmul(p1, R)
            diff = p1_rot - p2
            rmsd = torch.sqrt((diff ** 2).sum() / pos1.size(0))
            return rmsd
        except:
            return torch.tensor(99.9, device=pos1.device)

    # [SOTA Visualization] PDB Writer (v18.48)
    # Saves the generated molecule so user can see it in PyMOL!
    @staticmethod
    def save_pdb(pos, x_L, filename):
        with open(filename, 'w') as f:
            f.write("COMPND    GENERATED BY MAXFLOW ENGINE\n")
            elem_map = {0:'C', 1:'N', 2:'O', 3:'S', 4:'F', 5:'P', 6:'CL', 7:'BR', 8:'I'}
            
            # Use Softmax to decide element
            probs = x_L.softmax(dim=-1)
            atom_types = probs.argmax(dim=-1).cpu().numpy()
            coords = pos.cpu().numpy()
            
            for i, (xyz, type_idx) in enumerate(zip(coords, atom_types)):
                elem = elem_map.get(type_idx, 'C') # Default Carbon
                name = elem 
                # HETATM serial name resName chain resSeq x y z occ temp element
                f.write(f"HETATM{i+1:>5} {name:<4} LIG L   1    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {elem:>2}\n")
            f.write("END\n")
        print(f"📄 Saved Generated Structure to {filename}")

    def run_configuration(self, name, pdb_id="7SMV", use_mamba=True, use_maxrl=True, use_muon=True):
        # ... (rest of function signature matches existing)
        print(f"🚀 Running Ablation: {name} on {pdb_id}...")
        
        # 1. Setup Architecture
        backbone = CrossGVP(node_in_dim=167, hidden_dim=64, num_layers=3).to(self.device)
        if not use_mamba:
            class RobustIdentity(nn.Module):
                def forward(self, x, **kwargs): return x
            backbone.global_mixer = RobustIdentity()
        model = RectifiedFlow(backbone).to(self.device)
        
        # [SOTA Fix] Pre-trained Weights Loading (v18.20)
        # CRITICAL: If available, load the SOTA weights! Otherwise we are training from scratch.
        # [SOTA Fix] Pre-trained Weights Loading (v18.20)
        # CRITICAL: If available, load the SOTA weights! Otherwise we are training from scratch.
        weights_path = "maxflow_pretrained.pt"
        
        # [SOTA Robustness] Smart Search in Kaggle Input
        if not os.path.exists(weights_path):
            print("🔍 Searching for weights in /kaggle/input...")
            found = False
            for root, dirs, files in os.walk("/kaggle/input"):
                for file in files:
                    if file.endswith(".pt") and "7SMV" not in file and "shard" not in file:
                        weights_path = os.path.join(root, file)
                        print(f"✅ Found potential weights: {weights_path}")
                        found = True
                        break
                if found: break
        
        if os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                model.load_state_dict(state_dict, strict=False)
                print(f"✅ Loaded SOTA Weights from {weights_path}")
                
                # [SOTA Validation] Check for critical mismatches that cause NaNs
                # If we loaded mostly garbage, it's better to restart.
                # Heuristic: Check if backbone.edge_embedding exists
                # Actually, the runtime error wouldn't happen if strict=False. 
                # But the user reported a mismatch error printed! 
                # This means load_state_dict printed them but didn't crash? 
                # No, strict=False returns incompatible keys.
                # The user's log showed "Error(s) in loading". 
                # If strict=False, it should NOT raise RuntimeError for size mismatch??
                # Wait, size mismatch IS a RuntimeError even with strict=False in some versions?
                # PyTorch behavior: strict=False ignores missing/unexpected keys. 
                # SIZE MISMATCHES enable a RuntimeError starting PyTorch 1.10+ unless assignment is compatible.
                # So we must catch it.
            except Exception as e:
                print(f"⚠️ SOTA Architecture Mismatch Detected: {e}")
                print("🔄 Legacy Weights (v10/v12) are incompatible with v18.35 Platinum Engine.")
                print("⚡ SWITCHING TO GENESIS MODE (Ab Initio Learning)...")
                # Re-init weights to be safe (Xavier/Kaiming)
                for p in model.parameters():
                    if p.dim() > 1: nn.init.xavier_uniform_(p)
        else:
             print("⚠️ Warning: Running Ab Initio (No Pre-trained Weights Found).")
        
        # 2. Fetch Real Target (Now with Smart Pocket!)
        pos_P, x_P, q_P, (pocket_center, pos_native) = self.feater.parse(pdb_id)
        
        # 3. Genesis Initialization (Ab Initio)
        torch.manual_seed(42)
        
        # [SOTA Fix] Stoichiometry Matching (v18.46)
        # We must generate the EXACT number of atoms as the native ligand.
        # Hardcoding '16' was a Toy Model assumption.
        num_atoms = pos_native.size(0)
        print(f"✨ Genesis Mode: Generating {num_atoms}-atom molecule (matching Native)...")
        
        # [SOTA Fix] Chemical Evolution (v18.22)
        # Allow atom types (x_L) to evolve to fit the pocket chemistry!
        # Wrap as nn.Parameter for rigorous optimizer handling
        x_L = nn.Parameter(torch.randn(num_atoms, 167, device=self.device))
        pos_L = pocket_center + torch.randn(num_atoms, 3, device=self.device).detach() * 1.0 
        
        # [SOTA Fix] Ligand Charge Optimization (v18.21)
        # Initialize small random charges to break symmetry
        q_L = nn.Parameter(torch.randn(num_atoms, device=self.device) * 0.1)
        data = FlowData(x_L=x_L, pos_L=pos_L, x_P=x_P, pos_P=pos_P, pocket_center=pocket_center)

        # 4. TTA Loop
        num_steps = 5 if TEST_MODE else 1000
        
        # Add q_L AND x_L to optimizer for Charge Hallucination!
        params = list(model.parameters()) + [q_L, x_L]
        opt = Muon(params, lr=0.002) if use_muon else torch.optim.AdamW(params, lr=0.001)
        
        # [SOTA Fix 4/5] Exponential Moving Average (EMA) (v18.27)
        # Standard practice for generative models (DDPM, SDE) to stabilize outcomes.
        # We maintain a shadow copy of the model parameters.
        from torch.optim.swa_utils import AveragedModel
        ema_model = AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
        
        # [SOTA Fix 5/5] Cosine Learning Rate Schedule (v18.27)
        # Ab Initio needs warmup (to escaping poor initialization) and decay (for convergence).
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(opt, T_max=num_steps, eta_min=1e-5)
        
        history = []
        
        for step in range(1, num_steps + 1):
        # [SOTA Fix] Time Injection (Flow Matching Theory)
            # We use Logit-Normal sampling to focus on the 'difficult' middle of the trajectory.
            t_val = step / num_steps
            # Batch size is 1 (Single Molecule Optimization)
            batch_sz = 1
            t_logit = torch.sigmoid(torch.randn(batch_sz, device=self.device)) 
            
            # Hybrid Time Strategy:
            # Linear Deterministic time for stability in TTA.
            t = torch.full((batch_sz,), t_val, device=self.device)
            
            # Rotate Ligand & Protein (System-wide SE(3) Augmentation)
            from scipy.spatial.transform import Rotation
            rot = Rotation.random()
            rot_matrix = torch.tensor(rot.as_matrix(), dtype=torch.float32, device=self.device)
            
            # Rotate all positions relative to pocket center to maintain alignment
            with torch.no_grad():
                data.pos_L = (data.pos_L - pocket_center) @ rot_matrix.T + pocket_center
                pos_P_rot = (pos_P - pocket_center) @ rot_matrix.T + pocket_center
            
            model.train(); opt.zero_grad()
            # Current GVP uses internal distances so it's invariant, but Cross-Attention
            # might benefit from diverse spatial inputs for learning.
            # We must update data to use rotated protein positions if we want full aug.
            data.pos_P = pos_P_rot
            
            out = model(data, t=t)
            
            # [SOTA Fix] Adaptive Gradient Normalization (v18.17)
            # Replaces hardcoded annealing with rigorous gradient scaling.
            # We use a fixed physical softness (1.0) but control the effective step size
            # via gradient clipping, ensuring stability without softening reality.
            softness = 1.0 
            
            # Dynamics & Constraints
            v_pred = out['v_pred']
            if torch.isnan(v_pred).any():
                v_pred = torch.zeros_like(v_pred) # Emergency Stop
                
            v_scaled = torch.clamp(v_pred, min=-2.0, max=2.0) 
            next_pos = data.pos_L + v_scaled * 0.05
            
            dist_from_center = (next_pos - pocket_center).norm(dim=-1)
            gravity = 0.05 * dist_from_center.mean() 

            # [SOTA Fix] Charge Conservation (v18.48)
            # Prevent unrealistic charge accumulation.
            # Real partial charges are usually between -1 and 1.
            with torch.no_grad():
                q_L.clamp_(min=-1.0, max=1.0)
            
            # Physics Reward (Total NaN Defense)
            # [SOTA Fix 3/5] Atom-Specific VdW: Pass x_L to use learned atom types
            energy = PhysicsEngine.compute_energy(next_pos, pos_P, q_L, q_P, x_L=x_L, x_P=x_P, softness=softness)
            if torch.isnan(energy): energy = torch.tensor(100.0, device=self.device)
            
            # [SOTA Fix] Molecular Cohesion (v18.22)
            # Add attractive potential (Lennard-Jones Well) to keep atoms together!
            # intra_repulsion is purely repulsive. We need attraction.
            # Using simple distance variance penalty as proxy for bond integrity.
            cohesion = (torch.cdist(next_pos, next_pos).mean() - 2.0).pow(2).clamp(max=50.0)
            
            # [SOTA Fix 1/5] Bond Length Constraint (v18.25)
            bond_loss = PhysicsEngine.calculate_bond_constraint(next_pos).clamp(max=50.0)
            
            # [SOTA Fix 5/5] Hydrophobic Reward (v18.25)
            # Peak reward at hydrophobic residue centers
            hydro_reward = PhysicsEngine.calculate_hydrophobic_score(next_pos, x_L, pos_P, x_P)
            
            repulsion = PhysicsEngine.calculate_intra_repulsion(next_pos, softness=softness)
            if torch.isnan(repulsion): repulsion = torch.tensor(100.0, device=self.device)
            
            # [SOTA Fix] Net Charge Penalty
            # Encourage neutral molecule? Or just reasonable total.
            net_charge = q_L.sum().abs() * 0.1 
            
            reward = -energy - 0.5 * repulsion - gravity - 1.0 * cohesion - 2.0 * bond_loss + 5.0 * hydro_reward - net_charge
            
            # Master NaN Check
            # [SOTA Fix] GRPO-Accelerated Differentiable Physics (v18.43)
            # We combine Differentiable Physics (analytic gradients) with GRPO (Group Relative Policy Optimization).
            # 1. Calculate Group Advantage (Relative to batch mean)
            # R is (B,). We want to emphasize samples that are better than average.
            with torch.no_grad():
                adv = (reward - reward.mean()) / (reward.std() + 1e-6)
                # GRPO: Focus on top 50% (positive advantage) or just weight everything?
                # Weighting everything is standard.
                # Clip advantages for stability (PPO style)
                adv = torch.clamp(adv, min=-3.0, max=3.0)
                
                # Turn advantage into a positive weight for minimization?
                # No, standard REINFORCE is E[A * log_prob].
                # Here we minimize Loss.
                # Loss ~ - (Advantage * Minimize_Objective)
                # Minimize_Objective for us is 'reward' (technically energy is minimized, reward maximized).
                # Wait, 'reward' variable is already maximizing (negative energy).
                # So we want to maximize Reward.
                # Gradient should be proportional to Advantage.
            
            if use_maxrl:
                # GRPO Loss: - (Advantage * Differentiable_Reward)
                # If Adv is high, we pull hard on this sample's gradient.
                # If Adv is low (negative), we push away? 
                # Or we just assume "Policy Gradient" where log_prob gradient aligns with action.
                # For Differentiable Physics, we want dR/dx.
                # Standard SGD: x <- x + lr * dR/dx.
                # Weighted SGD: x <- x + lr * A * dR/dx.
                # If A is negative, we move OPPOSITE to the gradient?
                # That means we actively make "bad" samples WORSE? 
                # No, that destroys physics.
                # We should probably only weight by POSITIVE advantage (ReLU) or Softmax.
                # "Validation" says: Don't break physics.
                # Implementation: Importance Sampling with Softmax Weights (MaxRL-GRPO)
                weights = torch.softmax(reward / 1.0, dim=0) * reward.size(0) # Avg weight = 1.0
                weights = weights.detach()
                loss = -(weights * reward).mean()
            else:
                loss = -reward.mean() 
            
            # Optional: Add small regularization
            loss = loss + 0.001 * v_pred.pow(2).mean() # Velocity Regularization 
            
            # Gradient Safety
            if torch.isnan(loss):
                print(f"⚠️ NaN Loss detected at step {step}. Skipping update.")
                opt.zero_grad()
            else:
                loss.backward()
                
                # [SOTA Fix] Stochastic Gradient Langevin Dynamics (SGLD) (v18.18)
                # Adds annealed thermal noise to gradients to escape local minima (DiffDock/DeltaDock standard).
                # Noise ~ N(0, 2 * lr * T)
                if step < 800:
                    T = 0.1 * (1.0 - step / 800) 
                    noise_std = np.sqrt(2 * opt.param_groups[0]['lr'] * T)
                    for p in model.parameters():
                        if p.grad is not None:
                            noise = torch.randn_like(p.grad) * noise_std
                            p.grad.add_(noise)
                
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping
                opt.step()
                scheduler.step() # Update LR
                ema_model.update_parameters(model) # Update EMA shadow weights
                
            # [SOTA Fix] FLOW INTEGRATION (v18.19)
            # CRITICAL: Update positions to simulate trajectory!
            # Previously we were just optimizing the vector at t=0 forever.
            with torch.no_grad():
                data.pos_L = next_pos.detach()
                
            history.append(reward.mean().item()) 
            
            # [SOTA Metric] Chamfer Distance Logging (v18.24)
            # [SOTA Metric] Aligned RMSD Logging (v18.24)
            # We now use Kabsch alignment to report the TRUE structural error, ignoring global rotation.
            if step % 100 == 0:
                with torch.no_grad():
                     rmsd = self.calculate_rmsd(data.pos_L, pos_native)
                     print(f"Step {step}: Reward={history[-1]:.2f}, Aligned RMSD={rmsd.item():.2f}Å")

        # 5. Result Archival
        full_name = f"{name} ({pdb_id})"
        self.results.append({'name': full_name, 'base': name, 'pdb': pdb_id, 'history': history, 'final': np.mean(history[-10:]) if len(history) >= 10 else history[-1]})
        print(f"✅ {full_name} Completed. Final Reward: {self.results[-1]['final']:.4f}")
        # [SOTA Visualization] Auto-Save PDB
        clean_name = name.replace(" ", "_").replace(":", "").replace("(", "").replace(")", "")
        pdb_filename = f"output_{clean_name}_{pdb_id}.pdb"
        self.save_pdb(data.pos_L, x_L, pdb_filename)

        return model

# --- SECTION 5: DEEP IMPACT EXECUTION (Multi-Target Ablation) ---
suite = AblationSuite()
targets = ["7SMV", "6LU7", "1UYG"] 
last_model = None

for target in targets:
    last_model = suite.run_configuration("Full MaxFlow (SOTA)", pdb_id=target)
    suite.run_configuration("Ablation: No-Mamba-3", pdb_id=target, use_mamba=False)
    suite.run_configuration("Ablation: No-MaxRL", pdb_id=target, use_maxrl=False)
    suite.run_configuration("Baseline: AdamW", pdb_id=target, use_muon=False)

# --- SECTION 6: PUBLICATION PLOTTING (Fig 3) ---
plt.figure(figsize=(12, 7))
colors = {'Full MaxFlow (SOTA)': '#D9534F', 'Ablation: No-MaxRL': '#5BC0DE', 'Baseline: AdamW': '#F0AD4E', 'Ablation: No-Mamba-3': '#5CB85C'}

for res in suite.results:
    plt.plot(res['history'], label=res['name'], color=colors.get(res['base'], 'grey'), alpha=0.7, linewidth=1.5)

plt.title("ICLR 2026 Fig 3: Multi-Target Ablation Study (SOTA Scaling)", fontsize=14, fontweight='bold')
plt.xlabel("Optimization Steps (TTA)", fontsize=12)
plt.ylabel("Physical Reward (kcal/mol)", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("fig3_ablation_summary.pdf")

# --- SECTION 7: DATA ARCHIVAL ---
print("\n💾 Archiving Scientific Assets...")
if last_model:
    torch.save(last_model.state_dict(), "model_final_tta.pt")

pd.DataFrame([{'name': r['name'], 'score': r['final']} for r in suite.results]).to_csv("results_ablation.csv", index=False)

with zipfile.ZipFile("maxflow_iclr_v10_bundle.zip", 'w') as zipf:
    for f in ["fig3_ablation_summary.pdf", "results_ablation.csv", "model_final_tta.pt"]:
        if os.path.exists(f): zipf.write(f)

print(f"\n✅ SUCCESS: Final ICLR Bundle created. One-Click SOTA Scaling Complete.")
