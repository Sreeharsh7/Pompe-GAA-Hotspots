load 5NN3_criticality_in_bfactor.pdb, gaa
remove hydro
hide everything
show cartoon, polymer
bg_color white
color grey80, polymer
color 0x0072B2, b > 0.5 and b < 1.5
color 0xD55E00, b > 1.5 and b < 2.5
color 0x009E73, b > 2.5 and b < 3.5
color 0xE69F00, b > 3.5
show spheres, b > 0.5 and name CA
set sphere_scale, 0.9
orient
ray 2400, 2400
png fig3b_criticality.png, dpi=600
