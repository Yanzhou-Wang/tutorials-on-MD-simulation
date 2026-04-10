#!/bin/bash

wf="result-energy-vs-strain_NEP.txt"
cwd=`pwd`
r_dir="$cwd/../260227-2_*"

for i in `ls -d $r_dir/id-* |awk -F"/" '{print $NF}'`
do
	wd="$i";
	mkdir -p $wd
	rm -f  $wd/$wf
	for j in `ls -d $r_dir/$i/conv* |awk -F"/" '{print $NF}' |awk -F"_" '{print $2}' |sort -n |xargs`
	do
		strain=$j
		jn="conv_$j"
		n_sys=$(sed -n '1p' $r_dir/$i/$jn/model.xyz |awk '{print $1}')
		e_p=$(tail -n 1 $r_dir/$i/$jn/thermo.out | awk '{printf "%.6f\n", $3 / '$n_sys'}')
        ax=$(tail -n 1 $r_dir/$i/$jn/thermo.out | awk '{printf "%.6f\n", $10}')
        ay=$(tail -n 1 $r_dir/$i/$jn/thermo.out | awk '{printf "%.6f\n", $11}')
        az=$(tail -n 1 $r_dir/$i/$jn/thermo.out | awk '{printf "%.6f\n", $12}')
        bx=$(tail -n 1 $r_dir/$i/$jn/thermo.out | awk '{printf "%.6f\n", $13}')
        by=$(tail -n 1 $r_dir/$i/$jn/thermo.out | awk '{printf "%.6f\n", $14}')
        bz=$(tail -n 1 $r_dir/$i/$jn/thermo.out | awk '{printf "%.6f\n", $15}')
        cx=$(tail -n 1 $r_dir/$i/$jn/thermo.out | awk '{printf "%.6f\n", $16}')
        cy=$(tail -n 1 $r_dir/$i/$jn/thermo.out | awk '{printf "%.6f\n", $17}')
        cz=$(tail -n 1 $r_dir/$i/$jn/thermo.out | awk '{printf "%.6f\n", $18}')
volume=$(echo "$ax $ay $az $bx $by $bz $cx $cy $cz" | awk '
{
    ax=$1; ay=$2; az=$3;
    bx=$4; by=$5; bz=$6;
    cx=$7; cy=$8; cz=$9;

    vol = ax*(by*cz - bz*cy) - ay*(bx*cz - bz*cx) + az*(bx*cy - by*cx);
    if (vol < 0) vol = -vol;
    printf "%.6f\n", vol;
}')
        
        v_per_atom=$(echo $volume $n_sys |awk '{printf "%.2f\n", $1 / $2}') 
		echo "$strain	$v_per_atom     $e_p" | tee -a $wd/$wf
	done
    sed -i '1i#Strain	Volume(A^3/atom)    Energy(eV/atom)' $wd/$wf
	echo ">>>>>>>>>>>>>>>> $i .... <<<<<<<<<<<<<<<<<<<<<"
done
