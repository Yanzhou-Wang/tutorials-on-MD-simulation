#!/bin/bash
######## Attention ################################################################################
#######`dynasor` code sits in conda env, which needs to be activated beforce using it  ############
###################################################################################################

cwd=$(pwd)
for i in 3.5 		# ????????
do
        for j in 1	# sys size ???????
        do
		for k in 1 ; do    # cycles ??????
			writ_dire="job_${i}_$k"   #?????????????????????
               		mkdir -p $writ_dire
                	cd $writ_dire
#====dynasor input script as launcher ====================================================================
launcher=main_1-compute-ccf.py
traj="./dump.xyz"    ###???????????????
frame_stop=1000       ##????????
q_max=1
dt=10    # time interval between consective frames in traj
window_size=$(($frame_stop / 10))
window_step=$(($window_size / 10 ))
cat > $launcher <<!
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

trajectory_filename = '$traj'
traj = Trajectory(
    trajectory_filename,
    trajectory_format='extxyz',
    frame_stop=$frame_stop)

q_points = get_spherical_qpoints(traj.cell, q_max=$q_max, max_points=12000)

sample_raw = compute_dynamic_structure_factors(
    traj, q_points, dt=$dt, window_size=$window_size,
    window_step=$window_step, calculate_currents=True)


# the class save sample_raw 
with open('sample_raw.pkl', 'wb') as file:
    pickle.dump(sample_raw, file)

end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")
!
#=======================================================================================
#^^^^^^^^^ traj preparation ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

common_path="$cwd/../configure-files-for-example/example4"   #???????????????????
    ln -sf $common_path/$writ_dire/dump.xyz dump.xyz

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#### //////////  Prepare sbatch script  //////////////////////////////////////////////////////////////////////
#                	case_name=$(pwd |awk -F"/" '{printf "%s/%s", $'$(pwd |awk -F"/" '{print NF-1}')', $'$(pwd |awk -F"/" '{print NF}')'}')
			code=$launcher  # ???????????????????
#			cat > submit.sbatch <<!
##!/bin/bash
##SBATCH --nodes=1 --ntasks=20 		 ##12, 20 and 24. with --nodes, ntasks/nodes equals cores per node!!!!!!!!!!!!!!!!!!!!
##SBATCH --mem=64GB
##SBATCH --time=00-12:00:00                       # hh:mm:ss or dd-hh
##SBATCH --job-name="$case_name"
###SBATCH --mail-type=FAIL   --mail-user=yanzhowang@gmail.com          #BEGIN, END, FAIL, ALL.

#python $code
#!
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////

### !!!! submit job !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#		sbatch submit.sbatch; sleep 1s
export OMP_NUM_THREADS=10
python $code > cff.log
### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                	cd $cwd
               		echo "$i $j $k  done ..."
		done
        done
done
