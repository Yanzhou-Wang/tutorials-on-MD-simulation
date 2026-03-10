#!/bin/bash

# Usage: ./script.sh  # 在当前路径下查找运行abnormal的jobs，并把他们移入到当前目录的"ABNORMAL-JOB"里,并继承了他们原来的相对目录
#
# Assume abnormal jobs are organized as follows: 
# ./
# |__ id-mp-1191664-Ta5Ni  
# |__ id-mp-1218103-Ta4Cu3Ni9  
#      |__id-mp-1225687-CuNi
#         |__id-mp-1191669-Ta5Ni9 
#
#[OUTPUTs]:
# ./
# |__ABNORMAL-JOB
#    |__ id-mp-1191664-Ta5Ni  
#    |__ id-mp-1218103-Ta4Cu3Ni9  
#        |__id-mp-1225687-CuNi 
#           |__id-mp-1191669-Ta5Ni9

r_f="OUTCAR"

for i in $(find . -path "./ABNORMAL-JOB" -prune -o -name "$r_f" -print | sed 's#/'${r_f}'##')       # exclude from walking through ./ABNORMAL-JOB directory.
do
    job_path="$i"
    if [[ -z $(tail -n 25  ${job_path}/$r_f | grep "General timing and accounting informations for this job:") ]]
    then
        w_job_path=$(echo $job_path | sed 's#\./#\./ABNORMAL-JOB/#')
        w_job_parent=$(dirname "$w_job_path")
        mkdir -p $w_job_parent
        mv $job_path $w_job_parent
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        echo ">>>>>>>>> \"$job_path\" moved into \"$w_job_parent\" <<<<<<<<<<<<<<<<<"
        echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    else
        echo "$job_path is normally finished"
    fi
done
