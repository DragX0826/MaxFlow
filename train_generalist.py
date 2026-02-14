"""
MaxFlow Generalist Trainer (v18.35)
Trains a rigorous SOTA model on ANY dataset attached to /kaggle/input.

Usage:
    python train_generalist.py --epochs 100 --batch_size 16

Features:
    - Auto-detects .pt dataset files in /kaggle/input
    - Auto-detects pretrained weights (or starts from Genesis)
    - Uses Muon Optimizer + Logit-Normal Sampling (SOTA)
    - Saves checkpoints to /kaggle/working/checkpoints_generalist
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import glob
import sys
import subprocess
import random
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Batch

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
        safe_missing = [p for p in missing if p != 'meeko']
        if safe_missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + safe_missing)
        if 'meeko' in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "meeko", "--no-deps"])
    
    try:
        import torch_geometric
    except ImportError:
        print("🛠️  Installing Torch-Geometric (PyG)...")
        torch_v = torch.__version__.split('+')[0]
        cuda_v = 'cu' + torch.version.cuda.replace('.', '') if torch.cuda.is_available() else 'cpu'
        index_url = f"https://data.pyg.org/whl/torch-{torch_v}+{cuda_v}.html"
        pkgs = ["torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv", "torch-geometric"]
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + pkgs + ["-f", index_url])

auto_install_deps()

from maxflow.models.max_rl import MaxFlow
from accelerate import Accelerator

class GeneralistDataset(Dataset):
    def __init__(self, root_dirs):
        super().__init__()
        self.file_list = []
        if isinstance(root_dirs, str): root_dirs = [root_dirs]
        print(f"🔍 Scanning for datasets in {root_dirs}...")
        
        for r in root_dirs:
            if not os.path.exists(r): continue
            # Recursive scan for .pt files
            for root, dirs, files in os.walk(r):
                for file in files:
                    if file.endswith(".pt") and "checkpoint" not in file:
                        path = os.path.join(root, file)
                        self.file_list.append(path)
        
        print(f"✅ Found {len(self.file_list)} potential data shards.")
        if len(self.file_list) == 0:
            print("⚠️ No data found! Please attach a processed dataset (list of Data objects in .pt files) to /kaggle/input")

    def len(self):
        return len(self.file_list) * 100 # Dummy multiplier

    def get(self, idx):
        # Deterministic but randomized access
        path = random.choice(self.file_list)
        try:
            data = torch.load(path)
            if isinstance(data, list):
                return random.choice(data)
            return data
        except:
            return None # Handle corrupt files

def train_generalist(args):
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    os.makedirs(args.save_dir, exist_ok=True)
    
    accelerator.print(f"🚀 MaxFlow Generalist Training on {device}")
    
    # 1. Initialize Model
    try:
        # Standard SOTA config: 128 dim, 4 layers, Mamba-3
        student = MaxFlow(node_in_dim=167, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
        
        weights_path = None
        # Smart Weight Search
        if os.path.exists("maxflow_pretrained.pt"): weights_path = "maxflow_pretrained.pt"
        else:
             for root, dirs, files in os.walk("/kaggle/input"):
                for file in files:
                    if file.endswith(".pt") and "checkpoint" in file:
                        weights_path = os.path.join(root, file); break
        
        if weights_path:
            try:
                state_dict = torch.load(weights_path, map_location='cpu')
                student.load_state_dict(state_dict, strict=False)
                accelerator.print(f"✅ Loaded Base Weights from {weights_path}")
            except:
                accelerator.print("⚠️ Weight Mismatch. Restarting Genesis.")
                for p in student.parameters():
                    if p.dim() > 1: nn.init.xavier_uniform_(p)
        else:
            accelerator.print("✨ Genesis Mode: Initializing from Scratch (Xavier)")
            for p in student.parameters():
                if p.dim() > 1: nn.init.xavier_uniform_(p)
                
    except Exception as e:
        accelerator.print(f"⚠️ Initialization Error: {e}")
        return

    # 2. Optimizer
    try:
        from maxflow.utils.optimization import Muon
        optimizer = Muon(student.parameters(), lr=args.lr, momentum=0.95)
        accelerator.print("✅ Optimizer: Muon (SOTA)")
    except:
        optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
        accelerator.print("⚠️ Optimizer: AdamW (Fallback)")

    # 3. Data
    dataset = GeneralistDataset(["/kaggle/input", "./data"])
    if len(dataset.file_list) == 0:
        accelerator.print("❌ No data found. Exiting.")
        return

    # Custom collate because dataset returns single items that might be None
    def collate_safe(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0: return None
        return Batch.from_data_list(batch)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_safe)
    
    student, optimizer, loader = accelerator.prepare(student, optimizer, loader)
    
    # 4. Training Loop
    student.train()
    for epoch in range(args.epochs):
        total_loss = 0
        steps = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_local_main_process)
        
        for batch in pbar:
            if batch is None: continue
            
            optimizer.zero_grad()
            
            # [SOTA] Flow Matching Conditional Training
            # x_1 = Real Data (Ligand Positions)
            # x_0 = Noise
            
            x_1 = batch.pos_L
            x_0 = torch.randn_like(x_1)
            
            # t ~ Logit-Normal
            t = torch.sigmoid(torch.randn(batch.num_graphs, device=device))
            
            # Interpolate
            if hasattr(batch, 'batch'):
                 t_nodes = t[batch.batch].unsqueeze(-1)
            else:
                 t_nodes = t.reshape(-1, 1)

            x_t = t_nodes * x_1 + (1 - t_nodes) * x_0
            
            # We must condition on Protein (P) and Ligand Features (L)
            # The model expects a 'batch' object with pos_L modified
            batch_noisy = batch.clone()
            batch_noisy.pos_L = x_t
            
            # Predict Velocity
            # The target velocity is (x_1 - x_0)
            v_pred = student.policy(t, batch_noisy)['v_pred'] if hasattr(student, 'policy') else student(t, batch_noisy)['v_pred']
            v_target = x_1 - x_0
            
            loss = torch.nn.functional.mse_loss(v_pred, v_target)
            
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            pbar.set_postfix(loss=loss.item())
        
        # Save
        if accelerator.is_main_process and steps > 0:
            avg_loss = total_loss / steps
            accelerator.print(f"Epoch {epoch+1} Done. Loss: {avg_loss:.4f}")
            torch.save(accelerator.unwrap_model(student).state_dict(), 
                      f"{args.save_dir}/generalist_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden_dim", type=int, default=128) # SOTA standard
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="checkpoints_generalist")
    args = parser.parse_args()
    
    train_generalist(args)
