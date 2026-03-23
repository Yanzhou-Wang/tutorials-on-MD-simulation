# code block

```
# get voume/ion (A^3/atom) from OUTCAR
grep "volume\/ion" OUTCAR | tail -n 1 |awk '{print $5}'


## get number of ions from OUTCAR
n_sys=$(grep "number of ions" OUTCAR |awk '{print $12}' | tail -n 1)         


isol_ener=0
ener=$(grep "free  energy   TOTEN" OUTCAR | tail -1 | awk '{printf "%.6f\n", $5 - '$n_sys' * '$isol_ener'}')                #get energy from OUTCAR
ts=$(grep "T\*S" OUTCAR |tail -n 1 | awk '{printf("%12.6f\n", $5/'$n_sys')}')                                # get energy from OUTCAR


grep -A $((n_sys + 1)) "TOTAL-FORCE (eV/Angst)" $i | tail -n $n_sys |awk '{print $1,$2,$3,$4,$5,$6}' > result_position-force.txt               # get atomic position and corresponding forces from OUTCAR


NELM=$(grep -G "\<NELM\>" OUTCAR | awk -F';' '{print $1}' |awk '{print $3}')                # get NELM value from OUTCAR


grep -A 8 "TOTAL ELASTIC MODULI" OUTCAR |tail -n 6                                        # get elastic constants from OUTCAR

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
# Extract potential energy of system from OUTCAR
