#!/bin/bash

:<<!
]$ ls |awk -F"_" '{print $2}' |sort -n |xargs
0.95 0.96 0.97 0.98 0.99 1.00 1.01 1.02 1.03 1.04 1.05 1.06 1.07 1.08 1.09 1.10
!

wf="result-energy-vs-strain_DFT.txt"
cwd=`pwd`
r_dir="$cwd/../260227-1_*"

for i in `ls -d $r_dir/id-* |awk -F"/" '{print $NF}'`
do
	wd="$i";
	rm -rf $wd
	mkdir $wd
	for j in `ls -d $r_dir/$i/conv* |awk -F"/" '{print $NF}' |awk -F"_" '{print $2}' |sort -n |xargs`
	do
		strain=$j
		jn="conv_$j"
		n_sys=$(grep "number of ions" $r_dir/$i/$jn/OUTCAR |awk '{print $12}' | tail -n 1)
		e_p=$(grep "free  energy   TOTEN" $r_dir/$i/$jn/OUTCAR | tail -1 | awk '{printf "%.6f\n", $5 / '$n_sys'}')
        v_per_atom=$(grep "volume\/ion" $r_dir/$i/$jn/OUTCAR | tail -n 1 |awk '{print $5}')

		echo "$strain	$v_per_atom     $e_p" | tee -a $wd/$wf
	done
    sed -i '1i#Strain	Volume(A^3/atom) Energy(eV/atom)' $wd/$wf
	echo ">>>>>>>>>>>>>>>> $i .... <<<<<<<<<<<<<<<<<<<<<"
done
