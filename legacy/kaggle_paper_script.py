# kaggle_paper_script.py
"""
MaxFlow: Turnkey Kaggle Runner for ICLR 2027 Paper Results.
Usage in Kaggle:
!python kaggle_paper_script.py --smoke-test (for quick check)
!python kaggle_paper_script.py (for full results)
"""

import os
import json
import torch
import pandas as pd
from maxflow.tests.benchmark_sota_2026 import SOTABenchmark

def generate_paper_tables(results):
    print("\n📊 Generating Paper-Ready Tables...")
    
    # Flatten results for DataFrame
    data = []
    for mode, targets in results.items():
        for t_id, metrics in targets.items():
            row = {
                "Mode": mode,
                "Target": t_id,
                "Energy (kcal/mol)": round(metrics["energy"], 2),
                "Time (s)": round(metrics["time"], 2),
                "Status": metrics["status"]
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Summary Table (Mean per Mode)
    summary = df.groupby("Mode").agg({
        "Energy (kcal/mol)": ["mean", "std"],
        "Time (s)": ["mean", "std"]
    }).round(2)
    
    print("\n[Mean Results per Mode]")
    print(summary)
    
    # Export to Markdown for artifacts
    with open("paper_results_summary.md", "w") as f:
        f.write("# MaxFlow Paper Results Summary\n\n")
        f.write("## Summary Table\n\n")
        f.write(summary.to_markdown())
        f.write("\n\n## Detailed Results\n\n")
        f.write(df.to_markdown(index=False))
        
    # Export to LaTeX for the paper
    try:
        summary.to_latex("paper_results_table.tex")
        print("✅ LaTeX table saved to paper_results_table.tex")
    except Exception as e:
        print(f"⚠️ LaTeX export failed: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run a fast subset")
    args = parser.parse_args()
    
    print("🎗️ MaxFlow Kaggle Environment Check:")
    print(f"  - Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  - GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Initialize and run benchmark
    benchmark = SOTABenchmark()
    results = benchmark.run_eval(modes=["Baseline", "Full-SOTA"], smoke_test=args.smoke_test)
    
    # Generate tables
    generate_paper_tables(results)
    
    print("\n🚀 All done! You can now download paper_results_summary.md and paper_results_table.tex")
