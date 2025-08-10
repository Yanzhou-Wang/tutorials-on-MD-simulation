import numpy as np
#import os
from ase import Atoms
from ase.io import write

# Base structure of 4 atoms in fractional coordinates
r0 = np.array([
    [0.5,   0.0,   0.5],
    [0.0,   1/6,   0.5],
    [0.0,   0.5,   0.5],
    [0.5,   2/3,   0.5]
])
n0 = r0.shape[0]

# List of supercell sizes (n_x, n_y, n_z)
nxyzs = [
    [7, 4, 1],
    [35, 20, 1],
    [70, 40, 1],
    [105, 60, 1],
    [105, 20, 1],
    [140, 80, 1],
    [175, 100, 1],
    [210, 120, 1]
]

a = np.array([1.42 * np.sqrt(3), 1.42 * 3, 20.0])  # Lattice constants

for nxyz in nxyzs:
    nx, ny, nz = nxyz
    N = nx * ny * nz * n0
    positions = []
    tags_0 = []
    tags_1 = []

    center_nx = round(nx / 2 - 1)
    center_ny = round(ny / 2 - 1)
    tolerance = 200
    center_nx_lb = center_nx - tolerance
    center_nx_rb = center_nx + tolerance
    center_ny_lb = center_ny - tolerance
    center_ny_rb = center_ny + tolerance

    for iy in range(ny):
        for ix in range(nx):
            for iz in range(nz):
                for m in range(n0):
                    pos = a * (np.array([ix, iy, iz]) + r0[m])
                    positions.append(pos)

                    # tag_0: 1 for boundary atoms, 0 for interior atoms
                    if 0 < ix < nx - 1 and 0 < iy < ny - 1:
                        tags_0.append(0)
                    else:
                        tags_0.append(1)

                    # tag_1: 1 for central selected zone, 0 elsewhere
                    if (center_nx_lb <= ix < center_nx_rb) and (center_ny_lb <= iy < center_ny_rb):
                        tags_1.append(1)
                    else:
                        tags_1.append(0)

    positions = np.array(positions)
    group_tags = np.column_stack((tags_0, tags_1))  # shape (N, 2)

    atoms = Atoms('C' * len(positions), positions=positions)
    atoms.set_cell(a * nxyz)
    atoms.set_pbc([False, False, False])
    atoms.set_array("group", group_tags)

    model_name = f"model_size-{len(positions)}.xyz"
    write(model_name, atoms, format="extxyz")
    print(f"Generated {model_name} with {len(positions)} atoms")
