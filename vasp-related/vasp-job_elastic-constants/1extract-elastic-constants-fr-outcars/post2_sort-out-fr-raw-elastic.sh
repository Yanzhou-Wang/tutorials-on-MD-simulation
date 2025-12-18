#!/bin/bash
r_f="result-raw-elastic.txt"
w_f="result-sorted-elastic.txt"
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

echo "#job-id	C11	C12	C13	C22	C23	C33	C44	C55	C66 (GPa)" >> $w_f

for i in ${job_items[*]}
do
	c11=$(grep -A 6 $i $r_f |tail -n 6 | sed -n '1p' |awk '{print $1}')
	c12=$(grep -A 6 $i $r_f |tail -n 6 | sed -n '1p' |awk '{print $2}')
	c13=$(grep -A 6 $i $r_f |tail -n 6 | sed -n '1p' |awk '{print $3}')
	c22=$(grep -A 6 $i $r_f |tail -n 6 | sed -n '2p' |awk '{print $2}')
	c23=$(grep -A 6 $i $r_f |tail -n 6 | sed -n '2p' |awk '{print $3}')
	c33=$(grep -A 6 $i $r_f |tail -n 6 | sed -n '3p' |awk '{print $3}')
	c44=$(grep -A 6 $i $r_f |tail -n 6 | sed -n '4p' |awk '{print $4}')
	c55=$(grep -A 6 $i $r_f |tail -n 6 | sed -n '5p' |awk '{print $5}')
	c66=$(grep -A 6 $i $r_f |tail -n 6 | sed -n '6p' |awk '{print $6}')
	echo "$i  $c11  $c12  $c13  $c22  $c23  $c33  $c44  $c55  $c66" |tee -a tem.tem
done
column -t tem.tem >> $w_f
rm -f tem.tem
echo "Done!"
