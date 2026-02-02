#!/bin/bash
targ_file="OUTCAR"
MAINPATH=`pwd`
case_class=(.)    #???????????????????? Usually it is the current directory???
for index in ${case_class[*]}
do
        for i in `find $MAINPATH/$index -name "$targ_file" | sed 's/\/'$targ_file'//g'`
        do
                str1=$(echo $i | awk -F"/" '{print $(NF)}')     #Au.03.MP.5  #i=/scratch/work/wangy43/withUNEP/220728_InitStrc_18elements/3vasp/Au-done/Au.03.MP.5
                str2=$(echo $i | awk -F"/" '{print $(NF-1)}')   #Au-done
                str=$(echo $i | sed 's/\/'$str2'\/'$str1'//g')   #/scratch/work/wangy43/withUNEP/220728_InitStrc_18elements/3vasp
                writ_dire="$str/FAIL-SCF_$str2"
                mkdir -p $writ_dire

                dest="$i"
                if [[ `grep "General timing and accounting informations for this job" $dest/$targ_file` ]]; then
                        NELM=$(grep -G "^[ ]*[ ]NELM[ ][ ]*" $dest/$targ_file | tail -n 1 | awk -F';' '{print $1}' |awk '{print $3}')
                        electron_step=$(grep "^[ -][ -]*Iteration" $dest/$targ_file |tail -n 1 | awk '{print $4}'| awk -F')' '{print $1}')
                        if [[ $electron_step -eq $NELM ]]; then
                                echo "!!!!! $str2/$str1 NOT converged ... !!!!"
                                mv $dest $writ_dire
                        else
                                 echo "$str2/$str1 : scf successfully finished ..."
                        fi
                else
                        echo "! SOMETHING WRONG (the job fails to finish)! | $str2/$str1"
                        mv $dest $writ_dire
                fi

        	# if $writ_dire is empty, remove it
        	if [[ -d "$writ_dire" && -z "$(ls "$writ_dire")" ]]; then
                	rmdir "$writ_dire"
        	fi

        	done

done
