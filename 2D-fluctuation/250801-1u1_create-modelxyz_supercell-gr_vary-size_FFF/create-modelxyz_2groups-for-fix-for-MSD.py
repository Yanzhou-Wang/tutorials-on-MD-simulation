import numpy as np
#import os

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
    [140, 80, 1],
    [175, 100, 1],
    [210, 120, 1]
]

a = np.array([1.42 * np.sqrt(3), 1.42 * 3, 20.0])  # Lattice constants

for nxyz in nxyzs:
    nx, ny, nz = nxyz
    N = nx * ny * nz * n0
    r = np.zeros((N, 3))
    label_0 = np.zeros(N, dtype=int)
    label_1 = np.zeros(N, dtype=int)

    center_nx = round(nx / 2 - 1)
    center_ny = round(ny / 2 - 1)
    tolerance = 200
    center_nx_lb = center_nx - tolerance
    center_nx_rb = center_nx + tolerance
    center_ny_lb = center_ny - tolerance
    center_ny_rb = center_ny + tolerance

    n = 0
    for iy in range(ny):
        for ix in range(nx):
            for iz in range(nz):
                for m in range(n0):
                    r[n] = a * (np.array([ix, iy, iz]) + r0[m])

                    if 0 < ix < nx - 1 and 0 < iy < ny - 1:
                        label_0[n] = 0
                    else:
                        label_0[n] = 1

                    if (center_nx_lb <= ix < center_nx_rb) and (center_ny_lb <= iy < center_ny_rb):
                        label_1[n] = 1
                    else:
                        label_1[n] = 0

                    n += 1

    model_name = f"model_size-{N}.xyz"
    with open(model_name, 'w') as f:
        f.write(f"{N}\n")
        f.write(f"pbc=\"F F F\" Lattice=\"{a[0] * nx} 0 0 0 {a[1] * ny} 0 0 0 {a[2] * nz}\" ")
        f.write("Properties=species:S:1:pos:R:3:group:I:2\n")
        for i in range(N):
            f.write(f"C {r[i,0]:.6f} {r[i,1]:.6f} {r[i,2]:.6f} {label_0[i]} {label_1[i]}\n")

    print(f"Generated {model_name} with {N} atoms")
