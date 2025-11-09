#!/bin/bash
targ_file="OUTCAR"
find -maxdepth 2 -name "$targ_file" | sed 's/'$targ_file'//g' > result_job-index.dat
for i in $(find `pwd` -maxdepth 2 -name "$targ_file" | sed 's/\/'$targ_file'//g')   #?????????????
do
        dest=$i
        #job names are POSCAR.13  POSCAR.33  POSCAR.53  POSCAR.73, respectively. I need to get index of these jobs
        index=$(echo $dest | awk -F "/" '{print $NF}' | awk -F"." '{print $2}') #????????????????
        writ_file="result_magnetism_${index}.dat"
        n_sys=$(grep "number of ions" $dest/$targ_file |awk '{print $12}')
        magn_loca_line=$(($n_sys + 3))
        grep -A $magn_loca_line "magnetization (x)" $dest/$targ_file |tail -n $n_sys | awk '{print $5}' > $writ_file
        echo "$i done"
done
