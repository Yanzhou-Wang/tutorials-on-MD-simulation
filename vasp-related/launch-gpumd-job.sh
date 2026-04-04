#!/bin/bash
cwd=`pwd`
nep_dest="$HOME/PATH-TO-NEP-FILE"						# Mandatory Argument. "$nep_dest" specifies an exact path of nep file
code="$HOME/PATH-TO-GPUMD-CODE"							# Mandatory Argument, "$code" specifies an exact path of gpumd code							
stru_dir="$cwd/DIRE-TO-STRUCTURE"

is=(XX YY ZZ)                               			# Mandatory Argument, which is used to map structure name or job name
for i in ${!is[*]}
do
	js=(XX2 YY2 ZZ2)									# Mandatory Argument, which is 2nd-level, for mapping structure name or job name
	for j in ${!js[*]}
	do
		ks=(XX3 YY3 ZZ3)								# Mandatory Argument, which is 3rd-level, for mapping structure name or job name
		for k in ${!ks[*]} 
		do
			jn="JOB-NAME"								# Mandatory Argument, naming a jobname, probably $is[i], $js[j], $ks[k]-related 
			mkdir -p $jn
			cd $jn
			
			#model.xyz
			stru_n="STRU-FILE"							# Mandatory Argument, refering to structure name, probably $is[i], $js[j], $ks[k]-related 
			cp $stru_dir/$stru_n ./model.xyz
			
			#run.in										# Mandatory, for run.in generation
			cat > run.in << !
XXXXXXXXXXX
........
XXXXXXXXXX
!
			#sbatch script								# Mandatory, for job launching
			cat > submit-gpumd-job.sbatch <<!
YYYYYYYYY
.........
YYYYYYYYY
!
			sbatch submit-gpumd-job.sbatch
			cd $cwd
			echo ">>>>>>>>>>>>> ${is[i]} | ${js[j]} | ${ks[k]} <<<<<<<<<<<<<<<<<"
			sleep 1s
		done
	done
done
