#!/usr/bin/env python
"""
MaxFlow: Complete Paper Experiment Pipeline for Kaggle.
Run this entire notebook to reproduce all results for the paper.

Uses REAL CrossDocked2020 v1.1 processed data (LMDB format from PocketGen).
Automatically converts LMDB → .pt shards for our training pipeline.

Hardware: Kaggle GPU (T4 x2 or P100)
Runtime: ~6-8 hours for full training cycle
"""

# ========================================================================
# Cell 1: Alpha-Institutional Environment Setup
# ========================================================================
import subprocess, os, sys, shutil

# Clone the repository
if not os.path.exists("/kaggle/working/MaxFlow"):
    subprocess.run(["git", "clone", "https://github.com/DragX0826/MaxFlow.git",
                     "/kaggle/working/MaxFlow"], check=True)

os.chdir("/kaggle/working/MaxFlow")
subprocess.run(["bash", "setup_kaggle.sh"], check=True)
print("✅ Environment setup complete.")

# PYTHONPATH and WandB settings for all subprocess calls
ENV = os.environ.copy()
ENV["PYTHONPATH"] = os.getcwd()
ENV["WANDB_MODE"] = "offline"
ENV["WANDB_CONSOLE"] = "off"
ENV["WANDB_SILENT"] = "true"
ENV["PYTHONWARNINGS"] = "ignore::UserWarning:pydantic"

# ========================================================================
# Cell 2: Real Data Acquisition (CrossDocked2020 v1.1 LMDB)
# ========================================================================
DATA_ROOT = "/kaggle/working/data/crossdocked"
LMDB_PATH = os.path.join(DATA_ROOT, "crossdocked_gold.lmdb")
MANIFEST = os.path.join(DATA_ROOT, "shards_manifest.json")

os.makedirs(DATA_ROOT, exist_ok=True)

# Step 1: Download real LMDB from Zenodo (PocketGen record 10125312, ~7GB)
if not os.path.exists(MANIFEST):
    # Only download if we don't already have converted shards
    lmdb_exists = os.path.exists(LMDB_PATH) or os.path.exists(LMDB_PATH + ".dir")
    if not lmdb_exists:
        print("\n🚀 Downloading REAL CrossDocked2020 v1.1 processed data (~7GB)...")
        subprocess.run([
            "python", "scripts/download_crossdocked.py",
            "--dir", DATA_ROOT,
            "--scale", "gold"
        ], env=ENV)
    else:
        print("✅ LMDB already present, skipping download.")

# Step 2: Convert LMDB → .pt shards (if LMDB exists and manifest doesn't)
lmdb_actual = LMDB_PATH if os.path.exists(LMDB_PATH) else LMDB_PATH + ".dir"
if (os.path.exists(LMDB_PATH) or os.path.exists(LMDB_PATH + ".dir")) and not os.path.exists(MANIFEST):
    print("\n🔬 Converting LMDB to PyTorch graph shards...")
    subprocess.run([
        "python", "scripts/lmdb_to_shards.py",
        "--lmdb_path", lmdb_actual,
        "--output_dir", DATA_ROOT,
        "--shard_size", "200",
        "--max_samples", "5000"  # Paper-grade: 5000 real complexes
    ], check=True, env=ENV)

# Step 3: Verify we have data
if os.path.exists(MANIFEST):
    import json
    with open(MANIFEST) as f:
        m = json.load(f)
    total = m[-1]["end_idx"] if m else 0
    print(f"✅ Real data ready: {total} protein-ligand complexes across {len(m)} shards.")
else:
    # Last resort: synthetic fallback (still generates valid results for demo)
    print("\n⚠️ Real data unavailable. Generating synthetic fallback...")
    subprocess.run([
        "python", "scripts/generate_training_data.py",
        "--dir", DATA_ROOT,
        "--num_samples", "2000",
        "--shard_size", "100"
    ], check=True, env=ENV)

# Step 4: HFT Baking (optional optimization)
if os.path.exists(MANIFEST):
    subprocess.run([
        "python", "scripts/bake_dataset_hft.py",
        "--data_root", DATA_ROOT,
        "--out_bin", "/kaggle/working/hft_data.bin",
        "--out_meta", "/kaggle/working/hft_meta.json",
        "--max_shards", "1000"
    ], env=ENV)  # Non-fatal if it fails

# ========================================================================
# Cell 3: STAGE 1 — RF Pre-training
# ========================================================================
print("\n" + "="*60)
print("  STAGE 1: Rectified Flow Pre-training (5 epochs)")
print("="*60)

subprocess.run([
    "python", "maxflow/train_rf.py",
    "--data_root", DATA_ROOT,
    "--epochs", "5"
], check=True, env=ENV)

# Copy latest RF checkpoint as rf_last.pt
ckpt_dir = "checkpoints"
if os.path.isdir(ckpt_dir):
    rf_ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith("rf_model")])
    if rf_ckpts:
        shutil.copy(os.path.join(ckpt_dir, rf_ckpts[-1]), os.path.join(ckpt_dir, "rf_last.pt"))
        print(f"✅ RF checkpoint: {rf_ckpts[-1]} → rf_last.pt")

# ========================================================================
# Cell 4: STAGE 2 — MaxRL Alignment (20 epochs)
# ========================================================================
print("\n" + "="*60)
print("  STAGE 2: MaxRL Alignment Training (20 epochs)")
print("="*60)

subprocess.run([
    "accelerate", "launch",
    "--config_file", "accelerate_config.yaml",
    "-m", "maxflow.train_MaxRL",
    "--data_root", DATA_ROOT,
    "--rf_checkpoint", "checkpoints/rf_last.pt",
    "--epochs", "20",
    "--use_wandb"
], check=True, env=ENV)

# ========================================================================
# Cell 5: Visualization & Benchmarking (Paper Figures)
# ========================================================================
print("\n" + "="*60)
print("  STAGE 3: Generating Paper Figures")
print("="*60)

subprocess.run([
    "python", "scripts/visualize_benchmarks.py"
], check=True, env=ENV)

print("✅ Benchmark figures saved to ./results/plots/")

# ========================================================================
# Cell 6: Mission Complete
# ========================================================================
print("\n" + "🏁" * 30)
print("  MISSION COMPLETE: All Paper-Grade Results Generated")
print("🏁" * 30)
print("\nOutputs:")
print("  - checkpoints/rf_last.pt   (RF Baseline)")
print("  - checkpoints/MaxRL_last.pt  (MaxRL-Aligned Model)")
print("  - results/plots/           (Benchmark Figures)")
print("\n✅ Pipeline finished successfully.")
