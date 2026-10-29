load 5NN3_votes_in_bfactor.pdb, gaa
remove hydro
hide everything
show cartoon, polymer
bg_color white
spectrum b, blue_white_red, polymer and b > -0.5, minimum=0, maximum=5
set cartoon_transparency, 0
show sticks, polymer and resi 92+136+152+179+180+247+250+258+280+282+294+296+302+315+317+319+331+332+338+340+344+354+364+365+368+370+378+399+402+407+441+443+488+516+518+541+606+637+669+673+683+703+707+710+720+722+747+753+754+815+821+833 and not name N+C+O
set ray_opaque_background, 0
orient
ray 2400, 2400
png fig4b_votes.png, dpi=600
