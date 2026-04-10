#!/bin/bash

:<<!
(base) [wangyanzhou@c2ln5 ~/proj_Cu-Ta-Li-Ni-vasp-jobs/260202-2u2_extract-uniaxial-energy-vs-strain_DFT-NEP_fr260202-1-260202-2]$ ls ../260202-2_gpumd-single-point_uniaxial_alloy-simple_fr260202-1u1/
1copy-input-struc-here.sh      id-mp-10257_Ni       id-mp-1184069-CuNi   id-mp-1185312-LiCu3  id-mp-1225694-CuNi3  id-mp-135_Li
2structureFormat_vasp2exyz.py  id-mp-1059259_Cu     id-mp-1185204-Li3Ni  id-mp-1185338-LiCu3  id-mp-1225695-CuNi   id-mp-23_Ni
3launch-gpumd-job.sh           id-mp-1063005_Li     id-mp-1185214-Li3Ni  id-mp-1217756_Ta     id-mp-1225698-Cu3Ni  id-mp-2647044_Ta
id-mp-10173_Li                 id-mp-1184054-CuNi3  id-mp-1185266-LiNi3  id-mp-1225687-CuNi   id-mp-1246134_Ni     input-struc
(base) [wangyanzhou@c2ln5 ~/proj_Cu-Ta-Li-Ni-vasp-jobs/260202-2u2_extract-uniaxial-energy-vs-strain_DFT-NEP_fr260202-1-260202-2]$ ls ../260202-2_gpumd-single-point_uniaxial_alloy-simple_fr260202-1u1/id-mp-10173_Li/
orth_a_0.95  orth_a_1.00  orth_a_1.05  orth_a_1.10  orth_b_0.99  orth_b_1.04  orth_b_1.09  orth_c_0.98  orth_c_1.03  orth_c_1.08
orth_a_0.96  orth_a_1.01  orth_a_1.06  orth_b_0.95  orth_b_1.00  orth_b_1.05  orth_b_1.10  orth_c_0.99  orth_c_1.04  orth_c_1.09
orth_a_0.97  orth_a_1.02  orth_a_1.07  orth_b_0.96  orth_b_1.01  orth_b_1.06  orth_c_0.95  orth_c_1.00  orth_c_1.05  orth_c_1.10
orth_a_0.98  orth_a_1.03  orth_a_1.08  orth_b_0.97  orth_b_1.02  orth_b_1.07  orth_c_0.96  orth_c_1.01  orth_c_1.06
orth_a_0.99  orth_a_1.04  orth_a_1.09  orth_b_0.98  orth_b_1.03  orth_b_1.08  orth_c_0.97  orth_c_1.02  orth_c_1.07
!

cwd=`pwd`
r_dir="$cwd/../260229-2_*"

for i in `ls -d $r_dir/id-* |awk -F"/" '{print $NF}'`
do
	wd="$i";
	mkdir -p $wd
	for j in `ls -d $r_dir/$i/orth* |awk -F"/" '{print $NF}' |awk -F"_" '{print $2}' |sort -u |xargs`
	do
		
		wf="result-energy-vs-strain_${j}_NEP.txt"
		rm -f $wd/$wf
		for k in `ls -d $r_dir/$i/orth_${j}_* |awk -F"/" '{print $NF}' |awk -F"_" '{print $3}' |sort -n |xargs`
		do
			strain=$k
			jn="orth_${j}_${k}"
			n_sys=$(sed -n '1p' $r_dir/$i/$jn/model.xyz |awk '{print $1}')
			e_p=$(tail -n 1 $r_dir/$i/$jn/thermo.out | awk '{printf "%.6f\n", $3 / '$n_sys'}') 
			echo "$strain	$e_p" | tee -a $wd/$wf
		done
		sed -i '1i#Strain	Energy(eV/atom)' $wd/$wf
		echo ">>>>>>>>>>>>>>>> $i | $j .... <<<<<<<<<<<<<<<<<<<<<"
	done
done
