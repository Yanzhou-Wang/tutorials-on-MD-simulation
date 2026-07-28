# I. `run.in` generation

## II. single-point calc

```
cat > run.in << !
potential  ${nep_dest}

ensemble 	nve
time_step 	0
dump_thermo	1
dump_exyz   1
run  1
!
```
> GPUMD允许time_step=0这一极端情况。该情况下，意味着run=1后，粒子位置依然没有变，所以dump_thermo, dump_exyz等输出就是model.xyz对应的结果



# I. `sbatch` script

## II. DONG-FANG-HPC: GPU-based gpumd job

```
#!/bin/bash
code="/data/home/wangyanzhou/code_inst/GPUMD-4.8_260214/src/gpumd"

cat > submit-gpumd-job.sbatch <<!
#!/bin/sh
#SBATCH -N 1                    #1个节点
#SBATCH -n 1            #1个task. 在调用v100卡时，一个task默认分配3个cores
#SBATCH --ntasks-per-node=1     
#SBATCH --partition=v100      #v100/v100g32
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

ulimit -s unlimited
ulimit -l unlimited

module load cuda/12.5
$code
!
```
