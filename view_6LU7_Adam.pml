
# MaxFlow ICLR 2026 Visualization Script
load 6LU7.pdb, protein
load output_6LU7_Adam.pdb, ligand

# Visual Styling
hide everything
show cartoon, protein
set cartoon_transparency, 0.4
color lightblue, protein

show sticks, ligand
color magenta, ligand
set stick_radius, 0.2

# Focus on Interaction
select pocket, protein within 5.0 of ligand
show lines, pocket
color gray60, pocket

util.cbay ligand
zoom ligand, 8
bg_color white
set ray_opaque_background, on
