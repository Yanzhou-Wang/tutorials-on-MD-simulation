# shell blocks for GPUMD

```
n_sys=$(sed -n '1p' $i/model.xyz |awk '{print $1}')
ener_p=$(tail -n 1 $i/thermo.out | awk '{printf "%.6f\n", $3 / '$n_sys'}')
```
