#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from ase.io import read, write
from ase.constraints import FixAtoms

rf = "./slab_Cu-polydopamine.vasp"
wf = "./slab_Cu-polydopamine_frozen.vasp"

atoms = read(rf, format='vasp')

z_cut = 2.5  # Å

# mask=True 的原子会被固定（F F F）
mask_fixed = [(atom.symbol == 'Cu' and atom.position[2] < z_cut)
              for atom in atoms]

n_frozen = int(np.sum(mask_fixed))

atoms.set_constraint(FixAtoms(mask=mask_fixed))

# 写 VASP5 POSCAR；保持原子顺序；用笛卡尔坐标
write(wf, atoms, format='vasp', vasp5=True, direct=False, sort=False)

print(f"[INFO] Read:   {rf}")
print(f"[INFO] Total atoms: {len(atoms)}")
print(f"[INFO] Frozen Cu (z < {z_cut:.2f} Å): {n_frozen}")
print(f"[INFO] Wrote:  {wf}")
