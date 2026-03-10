#!/bin/bash
# Usage: ./script.sh  # 在当前路径下查找运行abnormal jobs, 或者normal但是scf不收敛的jobs，并把他们分别移入移当前目录的"ABNORMAL-JOB"，和FAIL-SCF-JOB目录。 并继承了他们原来的相对目录
#
# Assume abnormal jobs are organized as follows: 
# ./
# |__ id-mp-1191664-Ta5Ni  
# |__ id-mp-1218103-Ta4Cu3Ni9  
#      |__id-mp-1225687-CuNi
#         |__id-mp-1191669-Ta5Ni9 
#
#[OUTPUTs]:		#正常结束的jobs,但是scf不收敛
# ./
# |__FAIL-SCF-JOB
#    |__ id-mp-1191664-Ta5Ni  
#    |__ id-mp-1218103-Ta4Cu3Ni9  
#        |__id-mp-1225687-CuNi 
#           |__id-mp-1191669-Ta5Ni9
#
#[OUTPUTs]:
# ./
# |__ABNORMAL-JOB	#非正常结束的jobs.
#    |__ id-mp-119164-Cu5Ni  
#    |__ id-mp-121813-Ta4Cu3  
#        |__id-mp-122567-CuNi3 

r_f="OUTCAR"

for i in $(find . \( -path "./FAIL-SCF-JOB" -o -path "./ABNORMAL-JOB" \) -prune -o -name "$r_f" -print | sed 's#/'${r_f}'##') # exclude from walking through ./FAIL-SCF-JOB and ./ABNORMAL-JOB dirs.
do
	job_path="$i"
	if [[ -n $(tail -n 25  ${job_path}/$r_f | grep "General timing and accounting informations for this job:") ]]
	then
       		NELM=$(grep -G "^[ ]*[ ]NELM[ ][ ]*" ${job_path}/$r_f | tail -n 1 | awk -F';' '{print $1}' |awk '{print $3}')
                electron_step=$(grep "^[ -][ -]*Iteration" ${job_path}/$r_f |tail -n 1 | awk '{print $4}'| awk -F')' '{print $1}')
                
		if [[ $electron_step -eq $NELM ]]; 
		then
                	w_job_path=$(echo $job_path | sed 's#\./#\./FAIL-SCF-JOB/#')
	        	w_job_parent=$(dirname "$w_job_path")
   		     	mkdir -p $w_job_parent
        		mv $job_path $w_job_parent
        		echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        		echo ">>>>>>>>> FAIL-SCF-JOB:  \"$job_path\" moved into \"$w_job_parent\" <<<<<<<<<<<<<<<<<"
        		echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
                else
                        echo "SCF for $job_path is successfully finished"
                fi
                
	else
		w_job_path=$(echo $job_path | sed 's#\./#\./ABNORMAL-JOB/#')
		w_job_parent=$(dirname "$w_job_path")
		mkdir -p $w_job_parent
		mv $job_path $w_job_parent
		echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
		echo ">>>>>>>>> ABNORMAL-JOB: \"$job_path\" moved into \"$w_job_parent\" <<<<<<<<<<<<<<<<<"
		echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"

	fi
        	done
