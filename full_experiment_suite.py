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
        print(f"🛠️  Missing dependencies found: {missing}. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("✅ Dependencies Installed.")

auto_install_deps()

# --- SECTION 1: KAGGLE ENVIRONMENT SETUP ---
def setup_environment():
    print("🛠️  Authenticating Kaggle Workspace...")
    
    # 1. Aggressive Search for the folder CONTAINING 'maxflow'
    search_roots = ['/kaggle/working', '/kaggle/input', os.getcwd()]
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
class PhysicsEngine:
    @staticmethod
    def compute_energy(pos_L, pos_P, q_L, q_P, dielectric=80.0, softness=0.0):
        dist = torch.cdist(pos_L, pos_P)
        dist_elec = torch.sqrt(dist.pow(2) + softness)
        e_elec = (332.06 * q_L.unsqueeze(1) * q_P.unsqueeze(0)) / (dielectric * dist_elec)
        sigma = 3.5
        sigma_6 = sigma ** 6
        r6_eff = dist.pow(6) + softness
        term_r6 = sigma_6 / r6_eff
        e_vdw = 0.15 * (term_r6.pow(2) - 2 * term_r6)
        energy = (e_elec + e_vdw).clamp(min=-1000.0, max=1000.0).sum()
        return energy

    @staticmethod
    def calculate_intra_repulsion(pos, threshold=1.2, softness=0.0):
        if pos.size(0) < 2: return torch.tensor(0.0, device=pos.device)
        dist = torch.cdist(pos, pos) + torch.eye(pos.size(0), device=pos.device) * 10.0
        dist_eff = torch.sqrt(dist.pow(2) + softness)
        rep = torch.relu(threshold - dist_eff).pow(2).sum()
        return rep.clamp(max=1000.0)

# --- SECTION 3: REAL PDB FEATURIZER ---
class RealPDBFeaturizer:
    def __init__(self):
        from Bio.PDB import PDBParser
        self.parser = PDBParser(QUIET=True)
        self.aa_map = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19}

    def parse(self, pdb_id="7SMV"):
        path = f"{pdb_id}.pdb"
        if not os.path.exists(path):
            import urllib.request
            print(f"📥 Downloading target {pdb_id} from RCSB...")
            urllib.request.urlretrieve(f"https://files.rcsb.org/download/{path}", path)
        struct = self.parser.get_structure(pdb_id, path)
        coords, feats = [], []
        for model in struct:
            for chain in model:
                for res in chain:
                    if 'CA' in res and res.get_resname() in self.aa_map:
                        coords.append(res['CA'].get_coord())
                        one_hot = [0.0] * 21
                        one_hot[self.aa_map[res.get_resname()]] = 1.0
                        feats.append(one_hot)
        return torch.tensor(np.array(coords), dtype=torch.float32).to(device), torch.tensor(np.array(feats), dtype=torch.float32).to(device)

# --- SECTION 4: ABLATION RUNNER CORE (GRPO-STYLE) ---
class AblationSuite:
    def __init__(self):
        self.results = []
        self.device = device
        self.feater = RealPDBFeaturizer()
        
    def run_configuration(self, name, pdb_id="7SMV", use_mamba=True, use_maxrl=True, use_muon=True):
        print(f"🚀 Running Ablation: {name} on {pdb_id}...")
        
        # 1. Setup Architecture
        backbone = CrossGVP(node_in_dim=167, hidden_dim=64, num_layers=3).to(self.device)
        if not use_mamba:
            backbone.global_mixer = nn.Identity() 
        model = RectifiedFlow(backbone).to(self.device)
        
        # 2. Fetch Real Target
        pos_P, x_P = self.feater.parse(pdb_id)
        q_P = torch.zeros(pos_P.shape[0], device=self.device)
        pocket_center = pos_P.mean(0)

        # 3. Genesis Initialization (Ab Initio)
        torch.manual_seed(42)
        batch_size = 16
        x_L = torch.randn(batch_size, 167, device=self.device).detach()
        pos_L = pocket_center + torch.randn(batch_size, 3, device=self.device).detach() * 1.0 
        q_L = torch.zeros(batch_size, device=self.device).requires_grad_(True)
        data = FlowData(x_L=x_L, pos_L=pos_L, x_P=x_P, pos_P=pos_P, pocket_center=pocket_center)

        # 4. TTA Loop
        num_steps = 5 if TEST_MODE else 1000
        opt = Muon(model.parameters(), lr=0.002) if use_muon else torch.optim.AdamW(model.parameters(), lr=0.001)
        history = []
        
        for step in range(1, num_steps + 1):
            model.train(); opt.zero_grad()
            out = model(data)
            
            # Curriculum Softness (Alpha 200 -> 0.001)
            progress = max(0, min(1, (step - 100) / 700)) 
            softness = 200.0 * (1.0 - progress)
            if step > 800: softness = 0.001
            
            # Dynamics & Constraints
            v_scaled = torch.clamp(out['v_pred'], min=-5.0, max=5.0)
            next_pos = data.pos_L + v_scaled * 0.1
            dist_from_center = (next_pos - pocket_center).norm(dim=-1)
            gravity = 0.1 * dist_from_center.mean()

            # Physics Reward
            energy = PhysicsEngine.compute_energy(next_pos, pos_P, q_L, q_P, softness=softness)
            repulsion = PhysicsEngine.calculate_intra_repulsion(next_pos, softness=softness)
            reward = -energy - 0.5 * repulsion - gravity
            
            if use_maxrl:
                loss = maxrl_loss(out['v_pred'].mean(0), reward.mean(), torch.tensor(0.0, device=self.device))
            else:
                loss = -reward.mean() 
            
            loss.backward(); opt.step()
            history.append(reward.mean().item()) 

        # 5. Result Archival
        full_name = f"{name} ({pdb_id})"
        self.results.append({'name': full_name, 'base': name, 'pdb': pdb_id, 'history': history, 'final': np.mean(history[-10:]) if len(history) >= 10 else history[-1]})
        print(f"✅ {full_name} Completed. Final Reward: {self.results[-1]['final']:.4f}")
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
