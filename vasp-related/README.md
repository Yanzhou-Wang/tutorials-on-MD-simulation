# I. 结构文件格式转化
## II. `structureFormat_vasp2exyz-v3.py`
使用非常灵活，结构转化+超胞
支持的用法：

1) 当前目录递归查找，默认不超胞：   递归寻找POSCR, CONTCAR, 或xxx.vasp格式文件，转化后保存在对应的源目录
   ./structureFormat_vasp2exyz.py

2) 指定一个路径（目录或单文件），默认不超胞：    显示地给出文件名时，文件名即使不是POSCAR, CONTCAR, xxx.vasp时，也可以
   ./structureFormat_vasp2exyz.py path/to/dir
   ./structureFormat_vasp2exyz.py path/to/file

3) 指定源路径 + 超胞：
   ./structureFormat_vasp2exyz.py path/to/dir Nx Ny Nz
   ./structureFormat_vasp2exyz.py path/to/file Nx Ny Nz

4) 指定源路径 + 目标目录，默认不超胞：
   ./structureFormat_vasp2exyz.py path/to/src path/to/dst

5) 指定源路径 + 目标目录 + 超胞：
   ./structureFormat_vasp2exyz.py path/to/src path/to/dst Nx Ny Nz




# I. Relax lattice parameter (`VASP`)

- Workflow: 1) `VASP`优化晶胞/原包 --> 2) 使用脚本程序格式化优化后的晶胞/原包得到`conv.vasp` --> 3) 使用脚本程序，提取并表格化晶格常数:

1. `jobid=260224-1_IBRION1-optimize-Li`, `IBRION=2/1`优化原包或晶胞. 
2. `jobid=260224-1u2_identify-prim-conv-orth_Li_fr260224-1`, 使用`py_identify-vasp-stru_2_prim-conv-orth.py` identify 优化后的`CONTCAR`, 以获取对应的`prim.vasp`, `conv.vasp`, `orth.vasp`等
3. `jobid=260224-1u3_table-lattice-parameter_Li_fr260224-1u2`, 使用`py_table-lattice-para_fr-optimized-conv-cell.py` 提取并表格化晶格常数  




# I. Elastic constant (`VASP`)

- Workflow: 1) VASP优化晶胞/原包 --> 2) 使用脚本程序格式化优化后的晶胞/原包得到`conv.vasp` --> 3) IBRION=6计算优化后晶胞的弹性常数 --> 4) 脚本程序提取弹性常数:

1. `jobid=260224-1_IBRION1-optimize-Li`, `IBRION=2/1`优化原包或晶胞
2. `jobid=260224-1u2_identify-prim-conv-orth_Li_fr260224-1`, 使用`py_identify-vasp-stru_2_prim-conv-orth.py` identify 优化后的`CONTCAR`, 以
    获取对应的`prim.vasp`, `conv.vasp`, `orth.vasp`等
3. `jobid=260224-2_IBRION6-elastic-constants_Li_ksp0.1_fr260224-1u2`, 对优化好的`conv.vasp`晶胞，执行IBRION=6的弹性常数计算
4. `jobid=260224-3u1_table-elastic-constant_Li_fr260224-2`, 从`OUTCAR`中提取弹性常数信息



# I. EOS (`VASP` + `NEP`)

 - Workflow: 1) `VASP`优化晶胞/原包 --> 2) 使用脚本程序格式化优化后的晶胞/原包得到`conv.vasp` --> 3) 使用脚本程序生成三轴同比应变结构 --> 4) `VASP`/`NEP-GPUMD`单点计算 --> 5) 表格化数据并绘图

1. `jobid=260224-1_IBRION1-optimize-Li`, `IBRION=2/1`优化原包或晶胞 
2. `jobid=260224-1u2_identify-prim-conv-orth_Li_fr260224-1`, 使用`py_identify-vasp-stru_2_prim-conv-orth.py` identify 优化后的`CONTCAR`, 以
    获取对应的`prim.vasp`, `conv.vasp`, `orth.vasp`等
3. `jobid=260227-1u1_create-triaxial-strain-Li-struc-for-EOS_fr260224-1u2`, `1py1-1_gen_isotropic_strain_conv.py`读取`conv.vasp` 生成三轴同比应变结构
4. `jobid=260227-1_IBRION-1-scf_triaxial-EOS_Li_fr260227-1u1`, VASP单点计算； `jobid=260227-2_gpumd-single-point-triaxial-EOS_Li_fr260227-1u1`, GPUMD单点计算
5. `jobid=260227-3u2_table-plot-triaxial-EOS_Li_DFT-NEP_fr260227-1-260227-2`, 表格化DFT和NEP数据，并绘图


# I. Uniaxial deformation: potential energy vs. strain (`VASP` + `NEP`)

- Workflow: 1) `VASP`优化晶胞/原包 --> 2) 使用脚本程序格式化优化后的晶胞/原包得到`orth.vasp` --> 3) 使用脚本程序对称性分析并遍历非等价轴,生成单轴应变结构 --> 4) `VASP`/`NEP-GPUMD`单点计算 --> 5) 表格化数据并绘图

1. `jobid=260224-1_IBRION1-optimize-Li`, `IBRION=2/1`优化原包或晶胞
2. `jobid=260224-1u2_identify-prim-conv-orth_Li_fr260224-1`, 使用`py_identify-vasp-stru_2_prim-conv-orth.py` identify 优化后的`CONTCAR`, 以
    获取对应的`prim.vasp`, `conv.vasp`, `orth.vasp`等
3. `jobid=260229-1u1_create-uniaxial-strain-Li-struc_fr260224-1u2`, `1py1-1_gen_uniaxial_strain_orth.py`读取`orth.vasp`生成所有非等价轴的系列单轴应变结构
4. `jobid=260229-1_IBRION-1-scf_uniaxial_Li_fr260229-1u1`, VASP单点计算；`jobid=260229-2_gpumd-single-point_uniaxial_Li_fr260229-1u1`, GPUMD单点计算
5. `jobid=260229-3u1_table-plot-Li-uniaxial-energy-vs-strain_DFT-NEP_fr260229-1-260229-2`, 表格化DFT和NEP数据，并绘图







# I. AIMD (`VASP`)

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
