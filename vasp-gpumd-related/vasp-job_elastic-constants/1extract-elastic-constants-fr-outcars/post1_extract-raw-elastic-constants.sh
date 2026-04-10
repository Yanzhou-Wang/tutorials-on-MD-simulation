#!/bin/bash

r_dir="../251120-2_*"
w_f="result-raw-elastic.txt"
rm -rf $w_f

job_items=(
id-CuNi3_mp-1184054_primitive
id-Li3Ni_mp-1185204_primitive
id-mp-1101992-Ta2Ni
id-mp-1184069-CuNi
id-mp-1185214-Li3Ni
id-mp-1185266-LiNi3
id-mp-1185312-LiCu3
id-mp-1185338-LiCu3
id-mp-1187227-Ta3Cu
id-mp-1191664-Ta5Ni
id-mp-1225687-CuNi
id-mp-1225694-CuNi3
id-mp-1225695-CuNi
id-mp-1225698-Cu3Ni
id-mp-1867-Ta2Ni
id-mp-569776-TaNi3
id-mp-570491-TaNi3
id-mp-862658-LiCu3
id-mp-974058-LiCu3
id-mp-975882-Li3Cu
id-Ta4Cu3Ni9_mp-1218103_primitive
id-TaNi3_mp-891_primitive
)

for i in ${job_items[*]}
do
	echo "$i" | tee -a $w_f
	 grep -A 8 "TOTAL ELASTIC MODULI" $r_dir/$i/OUTCAR | tail -n 6 | awk '{printf "%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\n", $2/10, $3/10, $4/10, $5/10, $6/10, $7/10}' >> $w_f
done
echo "Done!"
