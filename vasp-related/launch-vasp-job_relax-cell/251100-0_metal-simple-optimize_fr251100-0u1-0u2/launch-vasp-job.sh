#!/bin/bash

cwd=$(pwd)
pot_dir="$cwd/../251100-0u1_agreed-vasp-inputs1/POTCARs"
r_dir1="$cwd/../251100-0u2_agreed-vasp-inputs2_poscars"
r_dir2="$cwd/../251100-0u1_agreed-vasp-inputs1"


items=($(ls -1 ${r_dir1}/*.vasp |awk -F"/" '{print $NF}' |awk -F"." '{print $1}'))

for i in ${items[*]}
do
	jb="jobid_$i"
	mkdir -p $jb
	
	cd $jb
	cp ${r_dir1}/${i}.vasp POSCAR
	cp ${pot_dir}/* .

	dos2unix POSCAR 2>/dev/null
	
	sed -n '6p' POSCAR |xargs cat  > POTCAR	
	rm -f Cu  Li  Ni  Ta
#	cp $r_dir2/INCAR .
	cat > INCAR <<!
LCHARG  =  .FALSE.
LWAVE  =  .FALSE.
ISTART = 0    #  job   : 0-new  1- orbitals from WAVECAR
ICHARG = 2    #  charge: 1-file 2-atom 10-const

KSPACING = 0.2
KGAMMA = .TRUE.
GGA = PE

ENCUT = 600
ALGO =  Normal    # alorithm for electron optimization, can be also FAST or ALL
NELM = 120        # of ELM steps, sometimes default is too small 
EDIFF = 1E-04
SIGMA = 0.02; 
  ISMEAR =  0   #! broadening in eV, -4-tet -1-fermi 0-gaus
PREC = Accurate
LREAL  =    A      # real space projection; slightly less accurate but faster 

NSW = 2000       # number of steps for IOM
IBRION = 2        # CG for ions, often 1 (RMM-DISS) is faster
ISIF = 3
EDIFFG =  -1E-02   # stopping-criterion for IOM (all forces smaller 1E-2)
POTIM  = 1      # step for ionic-motion (for MD in fs)


#  KPAR   =    4      # make 4 groups, each group working on one set of k-points 
#  NCORE  =    4      # one orbital handled by 4 cores 
!

	cat > submit-vasp-job.sbatch <<'!'
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
!

#	sbatch submit-vasp-job.sbatch
	sleep 2s
	cd $cwd
	echo ">>> \"$jb\" has been submitted ..."
done

