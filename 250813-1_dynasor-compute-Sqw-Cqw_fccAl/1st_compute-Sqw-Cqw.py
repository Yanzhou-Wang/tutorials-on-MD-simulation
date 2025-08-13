import numpy as np
#import matplotlib.pyplot as plt
#from ase import Atoms
from ase.build import bulk
from dynasor import compute_dynamic_structure_factors, Trajectory
from dynasor.qpoints import get_supercell_qpoints_along_path
from seekpath import get_path



trajectory_filename = 'dumpT300.NVT.atom.velocity.gz'
traj = Trajectory(trajectory_filename,
                  trajectory_format='lammps_internal', frame_stop=10000)
print(traj)



prim = bulk('Al', a=4.065)
path_info = get_path((
    prim.cell,
    prim.get_scaled_positions(),
    prim.numbers))
point_coordinates = path_info['point_coords']
path = path_info['path']
print(prim.cell)
print(point_coordinates)
print(path)



q_segments = get_supercell_qpoints_along_path(
    path, point_coordinates, prim.cell, traj.cell)
q_points = np.vstack(q_segments)
print(q_segments)
print(q_points)




sample = compute_dynamic_structure_factors(
    traj, q_points, dt=25.0, window_size=500,
    window_step=50, calculate_currents=True)
print(sample)
sample.available_correlation_functions



np.save('dat_q_segments.npy', np.array(q_segments, dtype=object), allow_pickle=True)
np.save('dat_path.npy',      np.array(path, dtype=object),      allow_pickle=True)
sample.write_to_npz('dat_sample.npz')



