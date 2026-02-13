import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_validation_results():
    # Load Data
    try:
        df = pd.read_csv('retrospective_validation_results.csv')
    except FileNotFoundError:
        print("⚠️ Results file not found. Run validation script first.")
        return

    # Visual Style
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 6))

    # --- Plot 1: Vina Score Distribution ---
    plt.subplot(1, 2, 1)
    
    # Filter valid scores
    valid_df = df.dropna(subset=['Vina'])
    
    # Color Palette: Clinical (Red), AI (Blue)
    palette = {"Clinical": "#e74c3c", "MaxFlow (AI)": "#3498db"}
    
    sns.violinplot(x='Source', y='Vina', data=valid_df, palette=palette, inner='quartile')
    sns.stripplot(x='Source', y='Vina', data=valid_df, color='black', alpha=0.3, jitter=True)
    
    # Add Benchmark Lines
    gc376_score = -8.5
    plt.axhline(y=gc376_score, color='green', linestyle='--', label=f'GC376 Threshold ({gc376_score})')
    
    plt.title('Binding Affinity Distribution (Lower is Better)')
    plt.ylabel('Vina Score (kcal/mol)')
    plt.legend()

    # --- Plot 2: QED vs Vina Scatter ---
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='Vina', y='QED', hue='Source', data=valid_df, palette=palette, s=100, alpha=0.8)
    
    # Add "Sweet Spot" Box
    plt.axvspan(-12, -8.5, ymin=0.6, ymax=1.0, color='green', alpha=0.1, label='SOTA Candidates')
    
    plt.title('Drug-likeness (QED) vs. Affinity')
    plt.xlabel('Vina Score (kcal/mol)')
    plt.ylabel('QED Score (0-1)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('validation_summary.png', dpi=300)
    print("✅ Validation Plot saved to 'validation_summary.png'")
    
    # Print Summary Table
    print("\n📊 Summary Statistics:")
    print(df.groupby('Source')[['Vina', 'QED', 'TPSA']].mean())

if __name__ == "__main__":
    plot_validation_results()
