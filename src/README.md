# SAEB-Flow: Shortcut-Accelerated Evolutionary Bayesian Flow

**SAEB-Flow** is a differentiable molecular docking framework combining:
- **Rectified Flow Matching** with Confidence-Bootstrapped Shortcuts (CBSF)
- **Differentiable Physics Engine** (standard LJ 12-6, Coulombic, solvation)  
- **Iterative Recycling** (AF2-style latent refinement)
- **Permutation-Invariant Backbone** (multi-head attention)

Benchmarked on the **Astex Diverse Set** (85 targets) and **DiffDock PDBbind-2020 timesplit** (362 targets).

---

## 🚀 Quick Start

### Local
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download Astex Diverse Set PDB files
python scripts/download_astex.py --output data/astex_pdb/

# 3. Run benchmark
python run_benchmark.py --bench_astex --pdb_dir data/astex_pdb/ \
    --steps 300 --batch_size 16 --mode inference --seed 42
```

### Astex-10 (FK-SMC + SOCM, v10.1)
```bash
# Canonical 10-target quick validation used for iteration loops
python run_astex10_fksmc_socm.py \
    --pdb_dir data/astex_pdb \
    --steps 300 --batch_size 16 \
    --seeds 42,43 \
    --output_dir results/astex10_fksmc_socm_v10_1
```

### Kaggle (T4 x2)
```bash
# Use the provided notebook: kaggle_notebook.ipynb
# Required datasets:
#   - astex-diverse   (upload from data/astex_pdb/ after running download_astex.py)
#   - saeb-flow-src   (this repo)

# Or run directly in a Kaggle notebook cell:
python run_astex10_fksmc_socm.py \
    --pdb_dir /kaggle/input/astex-diverse \
    --num_gpus 2 \
    --kaggle \
    --seeds 42,43
```

---

## 📁 Project Structure

```
src/
├── run_benchmark.py          # Main entry point
├── kaggle_notebook.ipynb     # Kaggle notebook
├── requirements.txt
├── saeb/
│   ├── core/
│   │   ├── model.py          # SAEBFlowBackbone (sinusoidal embed, Transformer)
│   │   └── innovations.py    # CBSF loss, shortcut step, recycling
│   ├── physics/
│   │   └── engine.py         # Differentiable force field (LJ 12-6, electrostatics)
│   ├── experiment/
│   │   ├── suite.py          # Training loop (Kabsch RMSD, LR warmup)
│   │   └── config.py         # SimulationConfig dataclass
│   ├── reporting/
│   │   └── visualizer.py     # 6 ICLR-grade publication figures
│   └── utils/
│       └── pdb_io.py         # PDB parsing + ESM-2 featurization
└── scripts/
    ├── download_astex.py     # Download Astex Diverse Set 85
    └── pack_results.py       # Package results/plots into timestamped zip
```

---

## 📊 Benchmark Datasets

| Dataset | Targets | Use |
|---------|---------|-----|
| [Astex Diverse Set](https://pubs.acs.org/doi/10.1021/jm060522a) | 85 | Standard re-docking evaluation |
| [DiffDock timesplit](https://github.com/gcorso/DiffDock) | 362 | Blind docking vs SOTA |

---

## 🔬 Key Metrics

- **SR@2A**: Success Rate at RMSD < 2Å
- **Median RMSD**: Kabsch-aligned, centroid-corrected
- **Energy**: Binding potential from differentiable force field

---

## 🔥 High-Performance Benchmarking (Kaggle)

For best results on Kaggle T4 x 2, our infrastructure supports:
- **Multi-GPU Parallelization**: Automatically utilizes both T4 GPUs via `torch.multiprocessing` (spawn).
- **High Fidelity Mode**: 1000 optimization steps + 64 batch sampling (`--high_fidelity`).
- **Multi-Seed Ensembling**: Run targets across multiple seeds for statistical robustness (`--seeds 42,43,44`).

```bash
# Example High-Performance Run
python run_benchmark.py --bench_astex --high_fidelity --seeds 42,43 --num_gpus 2
```

`run_benchmark.py` also supports custom subsets via:
```bash
python run_benchmark.py --targets 1aq1,1b8o,1cvu --fksmc --socm
```

---

## 📦 Packaging Results

```bash
python scripts/pack_results.py --label my_run_v1
# → saebflow_results_20260224_HHMMSS_my_run_v1.zip
```
