import os
import glob

def generate_pymol_script():
    """
    Generates 'load_results.pml' for PyMOL visualization.
    """
    pdb_files = glob.glob("*.pdb")
    output_pml = "load_results.pml"
    
    with open(output_pml, "w") as f:
        f.write("# PyMOL Visualization Script for MaxFlow v25.0\n")
        f.write("bg_color white\n")
        f.write("set ray_shadows, 0\n")
        f.write("set orthoscopic, 1\n")
        f.write("\n")
        
        for pdb in pdb_files:
            if "output_" not in pdb: continue
            
            # Load
            name = pdb.replace(".pdb", "")
            target = name.split("_")[-1] # output_Muon_7SMV -> 7SMV? No, name is output_Muon_7SMV
            
            f.write(f"load {pdb}, {name}\n")
            f.write(f"show sticks, {name}\n")
            f.write(f"color marine, {name} and elem C\n")
            f.write(f"util.ncb {name}\n")
            f.write("\n")
            
            # Try to load reference protein if exists
            # Convention: 7SMV.pdb
            # Parse target from filename: output_Muon_7SMV.pdb
            parts = name.split("_")
            if len(parts) >= 3:
                target_pdb = parts[-1] 
                f.write(f"# Loading Target {target_pdb}\n")
                f.write(f"fetch {target_pdb}, {target_pdb}_prot, async=0\n")
                f.write(f"remove solvent\n")
                f.write(f"show cartoon, {target_pdb}_prot\n")
                f.write(f"color gray80, {target_pdb}_prot\n")
                f.write(f"show surface, {target_pdb}_prot\n")
                f.write(f"set transparency, 0.7, {target_pdb}_prot\n")
                f.write(f"align {name}, {target_pdb}_prot\n")
                f.write(f"zoom {name}, 5\n")
                
        f.write("\n# Final View\n")
        f.write("center visible\n")
        f.write("zoom visible\n")
        
    print(f"✨ Generated {output_pml}. Run locally with 'pymol {output_pml}'")

if __name__ == "__main__":
    generate_pymol_script()
