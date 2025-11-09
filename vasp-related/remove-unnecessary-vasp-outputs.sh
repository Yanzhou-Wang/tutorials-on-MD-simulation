#!/bin/bash
rm_outs=(
CHG
CHGCAR
DOSCAR
EIGENVAL
PCDAT
WAVECAR
)

for i in ${rm_outs[*]}
do
        folder=$(date +%Y-%m-%d-%H:%M)
        mkdir -p ~/recycle/$folder
	find ./ -name "$i" -type f | xargs -n 1 -I{} mv {} ~/recycle/$folder/
	echo "===== $i removed ..."
done
