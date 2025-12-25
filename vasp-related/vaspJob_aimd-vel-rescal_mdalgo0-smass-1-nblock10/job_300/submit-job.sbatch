#!/bin/sh
#SBATCH -N 1                    #1个节点
#SBATCH -n 1            #1个task. 在调用v100卡时，一个task默认分配3个cores
#SBATCH --ntasks-per-node=1     
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

export vasp_path="/data/app/vasp/6.5.1-nvhpc"
module use /data/app/nvhpc/22.5_cuda12.9/modulefiles/
module load nvhpc-hpcx fftw/3.3.10-nvhpc
export PATH=${vasp_path}/bin:$PATH

ulimit -s unlimited
ulimit -l unlimited

mpirun -np $SLURM_NPROCS vasp_gam
