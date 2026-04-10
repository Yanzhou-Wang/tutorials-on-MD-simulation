#!/bin/bash
targ_file="OUTCAR"
find -maxdepth 2 -name "OUTCAR" | sed 's/'$targ_file'//g' > result_job-index.dat    # check here, might differ in specific situation ???????????
for i in $(find `pwd` -maxdepth 2 -name "$targ_file" | sed 's/\/'$targ_file'//g')   # get dest path for each OUTCAR?????????????
do
        #job names are POSCAR.13  POSCAR.33  POSCAR.53  POSCAR.73, respectively. I need to get index of these jobs
        index=$(echo $i | awk -F "/" '{print $NF}' | awk -F"." '{print $2}') #????????????????
        writ_file="result_atom_force_${index}.dat"
        dest=$i
        n_sys=$(grep "number of ions" $dest/$targ_file |awk '{print $12}')
        grep -A $((n_sys + 1)) "TOTAL-FORCE (eV/Angst)" $dest/$targ_file | tail -n $n_sys | awk '{print $4,$5,$6}' > $writ_file
        echo "$i done ..."
done
