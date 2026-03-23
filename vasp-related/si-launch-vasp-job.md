

# I. `INCAR` generation

## II. SCF calc for general metals

```
cat > INCAR <<!
# ===== Initialization =====
ISTART = 0
ICHARG = 2

# ===== K-points / XC =====
KSPACING = 0.2
KGAMMA = .TRUE.
GGA = PE

# ===== Electronic =====
ENCUT = 600
ALGO = Normal
NELM = 120
EDIFF = 1E-06
ISMEAR = 0
SIGMA = 0.02
PREC = Accurate

# ===== Ionic =====
NSW = 1
IBRION = -1

# ===== Output =====
LCHARG = .FALSE.
LWAVE  = .FALSE.

# ===== Performance =====
LREAL = A
!
```

## II. Elastic constant calc for general metals

```
cat > INCAR <<!
# ===== Initialization =====
ISYM = 2

# ===== Elastic / Ionic =====
IBRION = 6
ISIF = 3
NSW = 1
NFREE = 4
POTIM = 0.015
EDIFFG = -1E-2

# ===== Electronic =====
ENCUT = 600
PREC = Accurate
ALGO = Normal
NELM = 120
EDIFF = 1E-06
ISMEAR = 1
SIGMA = 0.1

# ===== XC =====
GGA = PE

# ===== Output =====
LCHARG = .FALSE.
LWAVE  = .FALSE.

# ===== Performance =====
LREAL = .FALSE.
KBLOWUP = .FALSE.
!
```



# I. sbatch script

## II. dongfang GPU-based VASP (vasp_std)
```
cat > submit-job.sbatch <<'!'
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

mpirun -np $SLURM_NPROCS vasp_std
!
```

