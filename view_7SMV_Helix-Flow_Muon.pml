
load 7SMV.pdb, protein
load output_7SMV_Helix-Flow_Muon.pdb, ligand
hide everything
show cartoon, protein
set cartoon_transparency, 0.4
color lightblue, protein
show surface, protein
set transparency, 0.7
set surface_color, gray80
show sticks, ligand
color magenta, ligand
set stick_size, 0.3
select pocket, protein within 5.0 of ligand
show lines, pocket
color gray60, pocket
util.cbay ligand
zoom ligand, 10
bg_color white
set ray_opaque_background, on
set antialias, 2
