# I. `run.in` generation

## II. single-point calc

```
cat > run.in << !
potential  ${nep_dest}

ensemble 	nve
time_step 	0
dump_thermo	1
run 		1
!
```


# I. sbatch script

## II. GPU-based gpumd job

```
cat > submit-gpumd-job.sbatch <<!
#!/bin/sh
#SBATCH -N 1                    #1个节点
#SBATCH -n 1            #1个task. 在调用v100卡时，一个task默认分配3个cores
#SBATCH --ntasks-per-node=1     
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

ulimit -s unlimited
ulimit -l unlimited

module load cuda/12.5
$code
!
```
