import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
import os

def generate_scientific_plots(results_path="all_results.pt"):
    """
    Generates ICLR-standard figures from experiment results.
    """
    if not os.path.exists(results_path):
        print(f"⚠️ {results_path} not found. Run full_experiment_suite.py first.")
        return

    results = torch.load(results_path)
    
    rows = []
    for res in results:
        # Load PDB properties
        qed, tpsa, mw = 0.0, 0.0, 0.0
        try:
            mol = Chem.MolFromPDBFile(f"output_{res['name']}.pdb")
            if mol:
                qed = QED.qed(mol)
                tpsa = Descriptors.TPSA(mol)
                mw = Descriptors.MolWt(mol)
        except: pass
        
        rows.append({
            "Method": "MaxFlow (Final)" if "Muon" in res['name'] else "AdamW Baseline",
            "Energy": res['final'],
            "RMSD": res['rmsd'],
            "QED": qed,
            "TPSA": tpsa,
            "MW": mw
        })
        
    df = pd.DataFrame(rows)
    
    # 1. Success Rate Statistics
    success_rate = (sum(1 for r in results if r['rmsd'] < 2.0) / len(results)) * 100
    print(f"\n🏆 [ICLR Metric] Success Rate (RMSD < 2.0A): {success_rate:.1f}%")
    
    # 2. Distribution Plots (Violin Plots)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    sns.set_theme(style="whitegrid", palette="Set2")
    
    # QED Distribution
    sns.violinplot(data=df, x="Method", y="QED", ax=axes[0], inner="stick")
    axes[0].set_title("Chemical Likeness (QED)")
    
    # Energy Distribution
    sns.violinplot(data=df, x="Method", y="Energy", ax=axes[1], inner="quart")
    axes[1].set_title("Binding Energy (kcal/mol)")
    
    # RMSD Distribution
    sns.violinplot(data=df, x="Method", y="RMSD", ax=axes[2], inner="point")
    axes[2].axhline(2.0, ls='--', color='red', label='Success Threshold')
    axes[2].set_title("Pose Recovery (RMSD)")
    axes[2].set_ylim(0, 10) # Clip outliers for visibility
    axes[2].legend()
    
    # [NEW] Multi-Target Variance
    sns.boxplot(data=df, x="Method", y="RMSD", hue="Method", ax=axes[3], showfliers=False)
    axes[3].set_title("Variance across Targets")
    
    plt.tight_layout()
    plt.savefig("fig2_publication_violin.png", dpi=300)
    print("✅ fig2_publication_violin.png saved.")
    
    # 3. Summary Table (LaTeX Ready)
    summary = df.groupby("Method").agg({
        "RMSD": ["median", "mean", "std"],
        "QED": "mean",
        "Energy": "mean"
    })
    print("\n📊 Quantitative Summary (ICLR Standard):")
    print(summary)
    summary.to_latex("table2_quantitative.tex")

if __name__ == "__main__":
    generate_scientific_plots()
