#!/bin/bash

cwd=$(pwd)
pot_dir="$cwd/PATH-TO-POT-DIR"        # Mandatory Argument. In the "$pot_dir", you must see POTCAR files forcbly named Cu, Ta, Li, Ni etc
stru_dir="$cwd/PATH-To-STRU-DIR"      # Mandatory Argument. $struc_dir refers where structure files for POSCAR are

is=(XX YY ZZ)                     # Mandatory Argument, which is used to map structure name or job name
for i in ${is[*]}
do
	js=(XX2 YY2 ZZ2)                # Mandatory Argument, which is secondary for mapping structure name or job name 
	for j in ${js[*]}
	do
		jn="interface_shift${i}_dist${j}"                    # Mandatory Argument, for building up job name
		
		mkdir -p $jn
		cd $jn
		
		#POSCAR
		stru_n="interface_shift${i}_dist${j}.vasp"          # Mandatory Argument, for refering to structure name for POSCAR
		cp ${stru_dir}/$stru_n POSCAR
	        
		#POTCAR
		cp ${pot_dir}/* .
        dos2unix -q POSCAR 
        sed -n '6p' POSCAR |xargs cat  > POTCAR       # Caveat: one's POTCAR files in ${pot_dir} MUST BE NAMED BY ELEMENT SYMBOLS, like Cu, Li, Ni, Ta etc.
        rm -f Cu  Li  Ni  Ta                          # Note: sync correspondingly

		#KPOINTS
		#ksp=0.2
		#$cwd/gen_kpoints_from_kspacing.py $ksp                                              # Option, used for generating KPOINTS file from specifed KSPACING  in INCAR
		#awk 'NR==4 {$3=1} {print}' KPOINTS > tmp && mv tmp KPOINTS                          # Option, one single gamma point along z-axis, usually used for calculations of surface or interface slab

		#INCAR
		cat > INCAR <<!                                                                      # Mandatory Argument, for generating INCAR
 XXXXXXXXXXXXXXXXX        
 ......
 XXXXXXXXXXXXXXXX
 !


		cat > submit-job.sbatch <<'!'                                                        # Mandatory Argument, for job launching
YYYYYYYYYYYYYYY
...............
YYYYYYYYYYYYYYY
!

		sbatch submit-job.sbatch
		sleep 1s
		cd $cwd

	done

done

