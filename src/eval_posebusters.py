import os
import argparse
import pandas as pd
from pathlib import Path
from posebusters import PoseBusters

def run_posebusters(pred_sdf, ref_sdf, protein_pdb):
    """
    Run PoseBusters check on a single target.
    Returns the full validation dataframe.
    """
    if not os.path.exists(pred_sdf) or not os.path.exists(ref_sdf) or not os.path.exists(protein_pdb):
        print(f"File missing: {pred_sdf}, {ref_sdf}, or {protein_pdb}")
        return None
        
    pdb = PoseBusters(config="dock")
    # For single target, pass as strings or ensure list handling is correct
    df = pdb.bust(pred_sdf, ref_sdf, protein_pdb)
    return df

def main():
    parser = argparse.ArgumentParser(description="Evaluate docking poses with PoseBusters")
    parser.add_argument("--pred", required=True, help="Predicted SDF file")
    parser.add_argument("--ref", required=True, help="Reference (native) SDF file")
    parser.add_argument("--prot", required=True, help="Protein PDB file")
    args = parser.parse_args()
    
    df = run_posebusters(args.pred, args.ref, args.prot)
    if df is not None:
        print("\n=== PoseBusters Audit Results ===")
        print(df.to_string())
        
        # Check if most standard tests passed
        cols = ["sanitization", "all_atoms_connected", "bond_lengths", "internal_steric_clash"]
        present_cols = [c for c in cols if c in df.columns]
        if present_cols:
            pass_rate = df[present_cols].mean(axis=1).iloc[0] * 100
            print(f"\nAverage Pass Rate (Key Checks): {pass_rate:.1f}%")

if __name__ == "__main__":
    main()
