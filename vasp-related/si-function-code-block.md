# I. `VASP`-related

```
#Extract NELM value from OUTCAR
NELM=$(grep -G "\<NELM\>" OUTCAR | awk -F';' '{print $1}' |awk '{print $3}')                # get NELM value from OUTCAR

#Extract voume/ion (A^3/atom) from OUTCAR
grep "volume\/ion" OUTCAR | tail -n 1 |awk '{print $5}'


#Extract number of ions from OUTCAR
n_sys=$(grep "number of ions" OUTCAR |awk '{print $12}' | tail -n 1)         


#Extract energy from OUTCAR
isol_ener=0
e_free=$(grep "free  energy   TOTEN" OUTCAR | tail -1 | awk '{printf "%.6f\n", $5 - '$n_sys' * '$isol_ener'}')                
e_ts=$(grep "T\*S" OUTCAR |tail -n 1 | awk '{printf("%12.6f\n", $5/'$n_sys')}')                                


#Extract atomic position and corresponding forces from OUTCAR
grep -A $((n_sys + 1)) "TOTAL-FORCE (eV/Angst)" $i | tail -n $n_sys |awk '{print $1,$2,$3,$4,$5,$6}' > result_position-force.txt  


#Extract elastic constant
grep -A 8 "TOTAL ELASTIC MODULI" OUTCAR |tail -n 6    




# Extract total magnetism from OUTCAR
targ_file="OUTCAR"
dest="."
n_sys=$(grep "number of ions" $dest/OUTCAR |awk '{print $12}')
magn_loca_line=$(($n_sys + 5))
totalM=$(grep -A $magn_loca_line "magnetization (x)" $dest/$n_sys |tail -n 1 | awk '{print $5}')
```
