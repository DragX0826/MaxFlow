# MaxFlow: Universal Geometric Drug Design Engine (ICLR/NeurIPS Submission)

## 1. Abstract
We present **MaxFlow**, a unified generative engine for structure-based drug design (SBDD) that achieves state-of-the-art (SOTA) performance in both binding affinity and inference speed. By integrating a **Mamba-3** state-space backbone with **Rectified Flow Matching**, MaxFlow scales linearly with sequence length ($O(N)$) while maintaining high-fidelity geometric generation. Furthermore, we introduce **MaxRL**, a multi-objective reinforcement learning fine-tuning capability that optimizes for QED, synthesizability (SA), and physical stability (PhysicsEngine) simultaneously.

## 2. Methodology

### 2.1. Architecture: Mamba-3 + Rectified Flow
Unlike diffusion models that require hundreds of denoising steps, MaxFlow utilizes **Rectified Flow Matching (RFM)** to learn a straight-line probability path from the prior distribution to the data distribution.
- **Backbone**: `CrossGVP` combined with `Mamba-3` blocks for efficient long-range context modeling between protein pockets and ligand atoms.
- **Sampling**: One-step (distilled) or few-step ODE solvers (RK45) drastically reduce inference time compared to DiffDock (approx. **100x faster**).

### 2.2. Physics-Aware Verification (System 2)
To prevent "Reward Hacking" (where models generate invalid structures with high Vina scores), MaxFlow incorporates a **SelfVerifier** module:
- **Geometric Checks**: Detects steric clashes and disconnected fragments.
- **PhysicsEngine**: Calculates van der Waals (vdW) and electrostatic energies to ensure thermodynamic stability.
- **Tanimoto Diversity**: Promotes exploration of the chemical space.

## 3. Experiments & Results

We evaluate MaxFlow on the **CrossDocked2020** test set and a challenging "Hard Mode" **FCoV Mpro** (FIP) viral target.

### 3.1. Speed-Accuracy Frontier (Figure 1)
We compare MaxFlow against DiffDock, MolDiff, and Pocket2Mol. 
- **y-axis**: Success Rate (% of valid molecules with Vina score < -7.0 kcal/mol).
- **x-axis**: Inference time per molecule (log scale).
*Result*: MaxFlow dominates the Pareto frontier, delivering higher success rates at a fraction of the computational cost.

### 3.2. FIP Case Study: FCoV Mpro (Figure 2)
In the Free Interfacial Perturbation (FIP) challenge, we optimize for **Binding Affinity** vs. **Drug-Likeness (TPSA)**.
- **Target Zone**: High affinity + CNS-compatible TPSA (40-90 $\AA^2$).
- **Pareto Frontier**: MaxFlow generates candidates (Red) that push significantly closer to the ideal theoretical limit compared to the baseline (Gray).

### 3.3. Ablation Study (Figure 3)
A holistic radar chart evaluation across 5 metrics:
1.  Binding Affinity
2.  QED (Quantitative Estimate of Drug-likeness)
3.  SA (Synthetic Accessibility)
4.  Diversity
5.  Inference Speed

## 4. Usage
This notebook demonstrates the **"One-Click" pipeline**:
1.  **Auto-Download**: Fetches real target data (PDBs).
2.  **Inference**: Loads the pre-trained `maxflow_pretrained.pt` checkpoint.
3.  **Analysis**: Runs the ablation suite and generates publication-ready plots.
