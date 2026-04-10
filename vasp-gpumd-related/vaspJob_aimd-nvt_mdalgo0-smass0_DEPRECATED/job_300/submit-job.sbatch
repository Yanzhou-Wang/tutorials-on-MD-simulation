#!/bin/sh
#SBATCH -N 1
#SBATCH -n 40
#SBATCH --ntasks-per-node=40
#SBATCH --partition=p1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

#module load vasp/6.5.0-intel
export PATH=/data/app/vasp-intel2024.2/6.5.0/bin:$PATH
source /data/app/intel/oneapi-2024.2/setvars.sh > /dev/null 2>&1

ulimit -s unlimited
ulimit -l unlimited
mpirun -np $SLURM_NPROCS vasp_gam
