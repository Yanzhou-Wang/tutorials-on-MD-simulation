#!/bin/bash
cwd=`pwd`
nep_dest="$HOME/PATH-TO-NEP-FILE"
code="$HOME/PATH-TO-GPUMD-CODE"
stru_dir="$cwd/DIRE-TO-STRUCTURE"

is=(XX YY ZZ)                               
for i in ${!is[*]}
do
	js=(XX2 YY2 ZZ2)
	for j in ${!js[*]}
	do
		ks=(XX3 YY3 ZZ3)
		for k in ${!ks[*]} 
		do
			jn="JOB-NAME"
			mkdir -p $jn
			cd $jn
			
			#model.xyz
			stru_n="STRU-FILE"
			cp $stru_dir/$stru_n ./model.xyz
			
			#run.in
			cat > run.in << !
XXXXXXXXXXX
........
XXXXXXXXXX
!
			#sbatch script
			cat > submit-gpumd-job.sbatch <<!
YYYYYYYYY
.........
YYYYYYYYY
!
			sbatch submit-gpumd-job.sbatch
			cd $cwd
			echo ">>>>>>>>>>>>> ${is[i]} ${js[j]} ${ks[k]} <<<<<<<<<<<<<<<<<"
			sleep 1s
		done
	done
done
