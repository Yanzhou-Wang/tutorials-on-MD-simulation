# extract data
```
volume=$(grep "volume of cell" OUTCAR |tail -n 1 |awk '{print $5}')                #get volume of box (A^3)
N=$(grep "number of ions" OUTCAR |tail -n 1 |awk '{print $12}')                       #get particle number of system
ener=$(grep "free  energy   TOTEN" OUTCAR | tail -1 | awk '{printf "%.6f\n", $5}')        #free energy
```


# Enumeate (all) vasp cases incuding OUTCAR
```
targ_file="OUTCAR"
for i in $(find `pwd` -maxdepth 2 -name "$targ_file" | sed 's/'$targ_file'//g')   #?????????????
do
        dest=$i
done
```
# Extract total magnetism from OUTCAR
```
targ_file="OUTCAR"
dest="."
n_sys=$(grep "number of ions" $dest/OUTCAR |awk '{print $12}')
magn_loca_line=$(($n_sys + 5))
totalM=$(grep -A $magn_loca_line "magnetization (x)" $dest/$n_sys |tail -n 1 | awk '{print $5}')
```
# Extract value of tag NELM from INCAR
```
NELM=$(grep -G "^[^#]*NELM" $dest/INCAR | head -n 1 | awk -F"=" '{print $2}' | awk '{print $1}')
NELM=$(grep -G "\<NELM\>" OUTCAR | awk -F';' '{print $1}' |awk '{print $3}')
```
# Extract atomic number of system from OUTCAR/POSCAR?
```
n_sys=$(sed -n '7p' POSCAR | awk '{for(i=1;i<=NF;i++){sum+=$i};print sum}')
n_sys=$(grep "number of ions" OUTCAR |awk '{print $12}')
```
# Extract all atom coordinations and corresponding forces of system from OUTCAR
```
n_sys=$(grep "number of ions" OUTCAR |awk '{print $12}')
grep -A $((n_sys + 1)) "TOTAL-FORCE (eV/Angst)" $i | tail -n $n_sys |awk '{print $1,$2,$3,$4,$5,$6}' > result_force.dat
```
# Extract potential energy of system from OUTCAR
```
n_sys=$(grep "number of ions" OUTCAR |awk '{print $12}')
isol_ener=0
ener=$(grep "free  energy   TOTEN" OUTCAR | tail -1 | awk '{printf "%.6f\n", $5 - '$n_sys' * '$isol_ener'}')
ts=$(grep "T\*S" OUTCAR |tail -n 1 | awk '{printf("%12.6f\n", $5/'$n_sys')}')
```

```
#!/bin/bash
wf="data-dft-FeCo.txt"
rm -f $wf
touch $wf

for i in `seq 0 1 14`
do  
    ener=$(grep "free  energy   TOTEN" $i/OUTCAR | tail -1 | awk '{printf "%.6f\n", $5}')    
    echo "$i    $ener" >> $wf
    echo "$i done ..." 
done
```
# Extract elastic constants

```
 grep -A 8 "TOTAL ELASTIC MODULI" OUTCAR |tail -n 6
```
