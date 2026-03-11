#!/bin/bash

# Usage: ./script.sh  	#在当前目录下递归遍历vasp的输出文件，删除用户指定的文件
# rm_outs=(xxx)数组中的文件名，如果以#开头，比如#vasprun.xml时，script.sh会排除对该文件的遍历和删除操作。
:<<!
CONTCAR
IBZKPT
INCAR
KPOINTS
OSZICAR
OUTCAR
POSCAR
POTCAR
REPORT
vasprun.xml
XDATCAR
!

rm_outs=(
CHG
CHGCAR
DOSCAR
EIGENVAL
PCDAT
IBZKPT
REPORT
#vasprun.xml
#XDATCAR
WAVECAR
)

for i in ${rm_outs[*]}
do
        # 如果以 # 开头，则跳过
        [[ $i == \#* ]] && continue

        folder=$(date +%Y-%m-%d-%H:%M)
        mkdir -p ~/recycle/$folder
		find ./ -name "$i" -type f -exec mv {} ~/recycle/$folder/ \;
        #find ./ -name "$i" -type f | xargs -n 1 -I{} mv {} ~/recycle/$folder/
        echo ">>>>>>>>>>> $i removed <<<<<<<<<<<<<<<<" 
done
