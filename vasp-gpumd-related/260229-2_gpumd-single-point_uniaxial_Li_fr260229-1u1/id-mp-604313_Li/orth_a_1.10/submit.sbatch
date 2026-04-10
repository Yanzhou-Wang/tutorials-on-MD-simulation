#!/bin/sh
#SBATCH -N 1                    #1个节点
#SBATCH -n 1            #1个task. 在调用v100卡时，一个task默认分配3个cores
#SBATCH --ntasks-per-node=1     
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load cuda/12.5

ulimit -s unlimited
ulimit -l unlimited

/data/home/wangyanzhou/code_inst/GPUMD-4.8_260214/src/gpumd
