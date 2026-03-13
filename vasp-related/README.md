# Elastic constant

- Workflow: 1)VASP优化晶胞/原包 --> 2)使用脚本程序，统一标准化获取优化后的晶胞 --> 3)IBRION=6计算优化后晶胞的弹性常数 --> 4)脚本程序提取弹性常数:

1. IBRION=2/1对原包或晶胞执行优化. `jobid=260224-1_IBRION1-optimize-Li`
2. 使用`py_identify-vasp-stru_2_prim-conv-orth.py` identify 优化后的`CONTCAR`, 以获取对应的`prim.vasp`, `conv.vasp`, `orth.vasp`等, `jobid=260224-1u2_identify-prim-conv-orth_Li_fr260224-1`
3. 对优化好的`conv.vasp`晶胞，执行IBRION=6的弹性常数计算. `jobid=260224-2_IBRION6-elastic-constants_simple-subs_ksp0.1_fr260224-1u2`
4. 从`OUTCAR`中提取弹性常数信息, `jobid=260224-3u1_get-elastic-constant_fr260224-2`





# I. AIMD

## II. NVE MD

- `vaspJob_aimd-nve_mdalgo0-smass-3`: NVE


## NVT MD

- `vaspJob_aimd-nvt_mdalgo0-smass0_DEPRECATED`: MDALGO=0, SMASS=0是vasp早期的nose-hoover thermostat的NVT实现
- `vaspJob_aimd-nvt_mdalgo2-smass0`: 后期发展的nose-hoover thermostat的NVT

> 二者的结果是相同的。早期版本的缺陷是轨迹XDATCAR的坐标被wraped into盒子内(分数坐标始终小于1），并且现在已经放弃维护了。因此不赞成使用了；
> 而后来的版本避免了坐标wrap，分数坐标可以大于1，这就可以直接基于轨迹文件来计算MSD或diffusion性质了。

## velocity-rescaling MD

- `vaspJob_aimd-vel-rescal_mdalgo0-smass-1-nblock1`: MDALGO=0和SMASS=-1同时存在时，二者是两种温度控制算法的一对矛盾。由于SMASS=-1的优先级高于IBRION=0, 所以体系选择SMASS=-1的紧致的速度标度的温控算法。也就是说，NBLOCK=1时，AIMD的温度始终被严格缩放T=TBEG + (TEND-TBEG)\*NSTEP/NSW. 
- `vaspJob_aimd-vel-rescal_mdalgo0-smass-1-nblock10`: NBLOCK=10,意味着第NSTEP=1，11，21 ...，步时采用了紧致的速度标度。之间的AIMD是微正则NVE系综模拟。
- `vaspJob_aimd-vel-rescal_mdalgo2-smass-1_ILL`: 病态的速度标度MD, MDALGO=2优先级较高，意味着使用nose-hoover thermstat算法， SMASS=-1又意味着速度标度算法。二者冲突，但由于MDALGO=2的优先级较高而导致SMASS=-1失效。所以模拟的结果，速度标失效。没有温度耦合，所以本质上是把体系近似看成了孤立体系的NVE模拟。

> `vaspJob_aimd-vel-rescal_mdalgo0-smass-1-nblock1`和`vaspJob_aimd-vel-rescal_mdalgo0-smass-1-nblock10`是正确的速度标度MD；而`vaspJob_aimd-vel-rescal_mdalgo2-smass-1_ILL`本质是NVE，是错误的速度标度
