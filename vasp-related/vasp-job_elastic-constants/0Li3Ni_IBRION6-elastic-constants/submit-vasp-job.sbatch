#!/bin/sh
#SBATCH -N 1
#SBATCH -n 40
#SBATCH --ntasks-per-node=40
#SBATCH --partition=p1
#SBATCH --output=%j.out
#SBATCH --error=%j.err
module load vasp/6.5.0-intel
ulimit -s unlimited
ulimit -l unlimited
mpirun -np $SLURM_NPROCS vasp_std
