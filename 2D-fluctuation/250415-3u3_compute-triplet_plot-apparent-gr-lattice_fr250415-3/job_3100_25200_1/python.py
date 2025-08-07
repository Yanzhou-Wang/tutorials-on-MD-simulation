'''
Activate ovito env before run the script:
    1) On triton: conda activate ovito-python3.11
'''

import os
import numpy as np
import ase.io
from ovito.io import import_file
from ovito.modifiers import CreateBondsModifier

from math import degrees, acos, sqrt
import time

# Input trajectory file path
input_trajectory = "/scratch/work/wangy43/work-master/Gr-fluc-gpumdJob/250415-3_graphene-lattice-vs-T_npt-scr_25200atom_denser-traj_6cycles_fr241013-1u/job_3100_25200_1/dump.xyz"
output_file = "bond.txt"

# Remove existing output file if it exists
if os.path.exists(output_file):
    os.remove(output_file)

# Read all frames using ASE
traj = ase.io.read(input_trajectory, index=slice(0,500,10))
total_frames = len(traj)

for frame_idx, frame in enumerate(traj):
    start_time = time.time()

    # Save current frame as temporary XYZ file
    temp_xyz = "temp_frame.xyz"
    ase.io.write(temp_xyz, frame, format="extxyz")

    # Load the frame into OVITO
    pipeline = import_file(temp_xyz)

    # Add bond creation modifier
    pipeline.modifiers.append(CreateBondsModifier(cutoff=2.0))

    # Compute data
    data = pipeline.compute()

    # Get positions, bond topology, and cell info
    positions = data.particles.positions
    bonds = data.particles.bonds.topology
    cell = data.cell
    pbc = cell.pbc
    cell_vectors = cell.matrix

    # Build neighbor list from bond topology
    neighbor_map = {}
    for i, j in bonds:
        neighbor_map.setdefault(i, []).append(j)
        neighbor_map.setdefault(j, []).append(i)

    # Compute bond lengths and angles per central atom
    triplet_data = []
    for center_idx, neighbors in neighbor_map.items():
        if len(neighbors) < 2:
            continue
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                a = positions[neighbors[i]] - positions[center_idx]
                b = positions[neighbors[j]] - positions[center_idx]

                # Apply PBC correction if necessary
                for d in range(3):
                    if pbc[d]:
                        a[d] -= round(a[d] / cell_vectors[d, d]) * cell_vectors[d, d]
                        b[d] -= round(b[d] / cell_vectors[d, d]) * cell_vectors[d, d]

                len_a = np.linalg.norm(a)
                len_b = np.linalg.norm(b)
                cos_theta = np.dot(a, b) / (len_a * len_b)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle = degrees(acos(cos_theta))

                # Compute third side (c) using cosine rule
                len_c = sqrt(len_a**2 + len_b**2 - 2 * len_a * len_b * cos_theta)
                triplet_data.append((len_a, len_b, angle, len_c))

    # Write results to output
    with open(output_file, "a") as f:
        for l1, l2, ang, l3 in triplet_data:
            f.write(f"{l1:.4f}\t{l2:.4f}\t{ang:.2f}\t{l3:.4f}\n")

    os.remove(temp_xyz)

    elapsed = time.time() - start_time
    print(f"Frame {frame_idx} processed in {elapsed:.2f} seconds.")

print(f"Analysis complete. Triplet bond lengths and angles written to {output_file}")
