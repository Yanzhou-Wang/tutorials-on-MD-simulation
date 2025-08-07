#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --ntasks=1		 ##n_node * n_core-per-node = ntasks

##SBATCH --nodes=1 
##SBATCH --ntasks=40		 ##n_node * n_core-per-node = ntasks

##SBATCH --nodes=2 
##SBATCH --ntasks=48		 ##n_node * n_core-per-node = ntasks

##SBATCH --ntasks-per-node=24     # 24, 28, 40, 128
#SBATCH --time="00-02:00:00"                       # hh:mm:ss or dd-hh
#SBATCH --job-name="250415-3u3_compute-triplet_plot-real-gr-lattice_fr250415-3/job_1000_25200_4"
#SBATCH --mem=64GB             # ???????????


#conda activate ovito-python3.11     # enable it just for calling for ovito-related modules on triton
# run code
srun python python.py
            

