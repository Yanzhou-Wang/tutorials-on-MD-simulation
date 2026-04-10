# shell blocks for GPUMD

```
n_sys=$(sed -n '1p' $i/model.xyz |awk '{print $1}')                        # get number of ions from model.xyz
ener_p=$(tail -n 1 $i/thermo.out | awk '{printf "%.6f\n", $3 / '$n_sys'}')  # get potential energy from thermo.out
#n_sys=$(sed -n '1p' $r_dir/$i/$j/model.xyz |awk '{print $1}')  
#e_slab=$(tail -n 1 $r_dir/$i/$j/thermo.out | awk '{printf "%.6f\n", $3 - '$n_sys' * '$p_e'}')


vecs=$(sed -n '2p' $r_dir/$i/$j/dump.xyz |awk -F"Lattice=\"" '{print $2}' |awk -F"\"" '{print $1}')    # get lattice vectors from target exyz structure.
a_vec=($(echo "$vecs" | awk '{print $1, $2, $3}'))
b_vec=($(echo "$vecs" | awk '{print $4, $5, $6}'))
```
