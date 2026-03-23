# I. `run.in` generation

## II. single-point calc

```
cat > run.in << !
potential  ${nep_dest}

ensemble 	nve
time_step 	0
dump_thermo	1
run 		1
!
```


# I. sbatch script

## GPU-based gpumd job

```
cat > run.in << !
potential  ${nep_dest}

ensemble 	nve
time_step 	0
dump_thermo	1
run 		1
!
```
