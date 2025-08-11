import numpy as np
import matplotlib.pyplot as plt
import pickle

from dynasor import compute_dynamic_structure_factors, Trajectory
from dynasor.qpoints import get_spherical_qpoints
from dynasor.post_processing import compute_spherical_qpoint_average

import time

start_time = time.time()

# set log level
from dynasor.logging_tools import set_logging_level
set_logging_level('INFO')

trajectory_filename = './dump.xyz'
traj = Trajectory(
    trajectory_filename,
    trajectory_format='extxyz',
    frame_stop=1000)

q_points = get_spherical_qpoints(traj.cell, q_max=1, max_points=12000)

sample_raw = compute_dynamic_structure_factors(
    traj, q_points, dt=10, window_size=100,
    window_step=10, calculate_currents=True)


# the class save sample_raw 
with open('sample_raw.pkl', 'wb') as file:
    pickle.dump(sample_raw, file)

end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")
