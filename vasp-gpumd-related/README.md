# TOC

- [I. VASP dataset generation and trained nep plots](#i-vasp-dataset-generation-and-trained-nep-plots)

- [I. Pre/ing/pos-process for GPUMD/VASP](#i-preingpos-process-for-gpumdvasp)
  - [II. Structure format conversion](#ii-structure-format-conversion)
    - [III. structure-format-conversion/structureFormat_vasp2exyz_v3.py](#iii-structure-format-conversionstructureformat_vasp2exyz_v3py)
    - [III. structure-format-conversion/structureFormat_exyz2vasp_v3.py](#iii-structure-format-conversionstructureformat_exyz2vasp_v3py)
    - [III. structure-format-conversion/structureFormat_cif2vasp_v3.py](#iii-structure-format-conversionstructureformat_cif2vasp_v3py)
  - [II. VASP-pre/ing/pos](#ii-vasp-preingpos)
  - [II. GPUMD-pre/ing/pos](#ii-gpumd-preingpos)

- [I. exe: VASP relaxes cell to get lattice parameter](#i-exe-vasp-relaxes-cell-to-get-lattice-parameter)

- [I. exe: VASP IBRION=6 computes elastic constants](#i-exe-vasp-ibrion6-computes-elastic-constants)

- [I. exe: OVITO computes and plot basic structure properties: coordination number, radial distribution function, bond length function, angular distribution function from restart.xyz](#i-exe-ovito-computes-and-plot-basic-structure-properties-coordination-number-radial-distribution-function-bond-length-function-angular-distribution-function-from-restartxyz)

- [I. exe: VASP/GPUMD computes EOS](#i-exe-vaspgpumd-computes-eos)

- [I. exe: VASP/GPUMD computes uniaxial deformation: potential energy vs. strain](#i-exe-vaspgpumd-computes-uniaxial-deformation-potential-energy-vs-strain)

- [I. VASP-AIMD (待处理)](#i-vasp-aimd-待处理)
  - [II. NVE AIMD](#ii-nve-aimd)
  - [II. NVT AIMD](#ii-nvt-aimd)
  - [II. velocity-rescaling MD](#ii-velocity-rescaling-md)




# I. VASP dataset generation and trained nep plots

-  `260101-1_outcar2xyz-dataset`:  Extract neccessary physical quantities from OUTCARs and build up exyz dataset that NEP code can read.

- `260101-3_split-dataset`:  Split single `dataset.xyz` into `train.xyz` and `test.xyz`

- `260101-5_trained-NEP_script-plottings`: plot NEP outputs, including `loss.out`, `energy_train.out`,  `force_train.out`,  `virial_train.out`, `energy_test.out` `force_test.out`, `virial_test.out`



# I. Pre/ing/pos-process for `GPUMD`/`VASP`

## II. Structure format conversion

### III. `structure-format-conversion/structureFormat_vasp2exyz_v3.py`

结构转化+超胞, 该脚本使用起来非常灵活，支持的用法：

1) 当前目录递归查找，默认不超胞：   递归寻找`POSCR`, `CONTCAR`, 或`xxx.vasp`格式文件，转化后保存在对应的源目录
   `./structureFormat_vasp2exyz.py`

2) 指定一个路径（目录或单文件），默认不超胞：    显示地给出文件名时，文件名即使不是POSCAR, CONTCAR, xxx.vasp时，也可以
   `./structureFormat_vasp2exyz.py path/to/dir`
   `./structureFormat_vasp2exyz.py path/to/file`

3) 指定源路径 + 超胞：
   `./structureFormat_vasp2exyz.py path/to/dir Nx Ny Nz`
   `./structureFormat_vasp2exyz.py path/to/file Nx Ny Nz`

4) 指定源路径 + 目标目录，默认不超胞：
   `./structureFormat_vasp2exyz.py path/to/src path/to/dst`

5) 指定源路径 + 目标目录 + 超胞：
   `./structureFormat_vasp2exyz.py path/to/src path/to/dst Nx Ny Nz`




### III. `structure-format-conversion/structureFormat_vasp2exyz_v3.py`

  1) 当前目录递归查找：
     递归寻找 *.xyz, *.exyz, *.extxyz 文件，
     将其中每一个结构帧分别转换为一个 .vasp 文件，
     输出到对应源目录下：
     ./structureFormat_exyz2vasp.py

  2) 指定一个路径（目录或单文件）：
     ./structureFormat_exyz2vasp.py path/to/dir
     ./structureFormat_exyz2vasp.py path/to/file.xyz

  3) 指定源路径 + 目标目录：
     ./structureFormat_exyz2vasp.py path/to/src path/to/dst
  4) It can also handle a dataset that includes many exyz frames
      ./structureFormat_exyz2vasp.py ./dataset.xyz ./DIR


### III. `structure-format-conversion/structureFormat_cif2vasp_v3.py`

Usage:

1) Recursively convert all `*.cif` under current directory (in-place) and write it in same place where x.cif is:
     `./py_cif2vasp.py`
     `./py_cif2vasp.py ./`

2) Convert under a given path (in-place) and write it in same place:
     `./py_cif2vasp.py /path/to/root_dir`
     `./py_cif2vasp.py /path/to/file.cif`

3) Convert from READ_PATH and write outputs into WRITE_PATH:
     `./py_cif2vasp.py read_path write_path`
   
   - If read_path is a `.cif` file:
       `write_path/<basename>.vasp`
   - If read_path is a directory:
       recursively find `**/*.cif` under read_path, and write to write_path
       keeping relative subdirectories (to avoid name conflicts)
   4) `../py_cif2vasp.py ../Ta2Ni_mp-1101992_primitive.cif ./`

Output style (same as your original):

1. `write(..., format="vasp", vasp5=True, direct=True, sort=False)`



## II. VASP-pre/ing/pos

### III. 固定POSCAR文件中z坐标方向的某些原子

- 固定POSCAR文件中z坐标方向的某些原子： `freeze-vasp-stru-z-dir-atoms/py_freeze-vasp-atoms.py`


### III. 由`KSPACING=0.2`和`POSCAR`生成`KPOINTS`文件

- 由`KSPACING=0.2`和`POSCAR`生成`KPOINTS`文件：  `generate-KPOINTS-fr-kspacing-POSCAR/gen_kpoints_from_kspacing.py`

### III. `launch-vasp-job.sh` 提交`vasp`作业

- 提交`vasp`作业： `./launch-vasp-job.sh`, .`/si-launch-vasp-job.md`

### III. 绘图原子受力，应力随离子步变化曲线

- 查看离子优化/盒子优化的力，应力随步数收敛： `plot-force-stress-gforce-converg-vs-ionic-step_fr-vasprunxml/py_plot-vasprun.xml_force-stress-gforce-converg-vs-ionic-step_v3.py`

#### III. 找到晶胞结构的空间群，晶系，结构类型

- 找到晶胞结构的空间群，晶系，结构类型： `260224-1u4_identify-table_space-group_crystal-system_stru-type/py_identify_space-group_crystal-system_stru-type_v3.py`





## II. GPUMD-pre/ing/pos

### III. 结构`model.xyz`建模

- 建模金刚石： `260410-0u1_create-model.xyz_diamond-vary-density/py_create-C-diamond-by-density.py`. Create crystal diamond varying densities, and varying supercell sizes

### III. `launch-gpumd-job.sh`提交作业

- 提交`gpumd`作业： `./launch-gpumd-job.sh`
- 支撑内容： `./si-launch-gpumd-job.md`

### III. 绘图`thermo.out`

- 绘图`thermo.out`: `thermo.out/py_plt-thermo-out_v4.py`. The script plots thermal quanties versus production time in `thermo.out`, including temperature T(t), potential energy U(t), pressure P(t), and lattice parameter curves.

### III. 绘图`kappa.out`

-  `plot-kappa.out-shc.out/py01_plt-kappa-vs-t_all-single-cycle-_fr-kappa.out_v1.py`, 绘图每个算例的每一次的 kappa vs time曲线 
-  `plot-kappa.out-shc.out/py02_plt-kappa-vs-t_all-averaged-n-cycle_fr-kappa.out_v1.py`， 绘图每个算例多次平均的kappa vs time 曲线


### III. 绘图`shc.out` 

-  `plot-kappa.out-shc.out/py11_plt-Kt-kw_all-single-cycle_fr-shc.out.py`, 绘图每个算例每一次的 $K(t)$, `$k(\omega)$` $E=mc^2$

-  `plot-kappa.out-shc.out/py12_plt-kw_all-averaged-n-cycle_fr-shc.out.py`, 绘图每个算例多次平均的`$k(\omega)$`
-  `plot-kappa.out-shc.out/py13_plt-kw-kwq_all-averaged-n-cycle_fr-shc.out.py`, 绘图多次平均的有量子校正的`$k(\omega)^q$`





# I. exe: `VASP` relaxes cell to get lattice parameter

- Workflow: 1) `VASP`优化晶胞/原包 --> 2) 使用脚本程序格式化优化后的晶胞/原包得到`conv.vasp` --> 3) 使用脚本程序，提取并表格化晶格常数:
1. `jobid=260224-1_IBRION1-optimize-Li`, `IBRION=2/1`优化原包或晶胞. 
2. `jobid=260224-1u2_identify-prim-conv-orth_Li_fr260224-1`, 使用`py_identify-vasp-stru_2_prim-conv-orth.py` identify 优化后的`CONTCAR`, 以获取对应的`prim.vasp`, `conv.vasp`, `orth.vasp`等
3. `jobid=260224-1u3_table-lattice-parameter_Li_fr260224-1u2`, 使用`py_table-lattice-para_fr-optimized-conv-cell.py` 提取并表格化晶格常数  



# I. exe: `VASP` `IBRION=6` computes elastic constants

- Workflow: 1) VASP优化晶胞/原包 --> 2) 使用脚本程序格式化优化后的晶胞/原包得到`conv.vasp` --> 3) IBRION=6计算优化后晶胞的弹性常数 --> 4) 脚本程序提取弹性常数:
1. `jobid=260224-1_IBRION1-optimize-Li`, `IBRION=2/1`优化原包或晶胞
2. `jobid=260224-1u2_identify-prim-conv-orth_Li_fr260224-1`, 使用`py_identify-vasp-stru_2_prim-conv-orth.py` identify 优化后的`CONTCAR`, 以
    获取对应的`prim.vasp`, `conv.vasp`, `orth.vasp`等
3. `jobid=260224-2_IBRION6-elastic-constants_Li_ksp0.1_fr260224-1u2`, 对优化好的`conv.vasp`晶胞，执行`IBRION=6`的弹性常数计算. 要使弹性常数计算作业不报错，要把`INCAR`里的`KSPACING`参数替换成`KPOINTS`, 完成这一任务的脚本程序`gen_kpoints_from_kspacing.py`放在了`generate-KPOINTS-fr-kspacing-POSCAR`目录里.
4. `jobid=260224-3u1_table-elastic-constant_Li_fr260224-2`, 从`OUTCAR`中提取弹性常数信息







# I. exe: `OVITO` computes and plot basic structure properties: coordination number, radial distribution function, bond length function, angular distribution function from `restart.xyz`

- 计算rdf: `260410-1u1_ovito-compute-plot_cn-rdf-adf_fr260410-1/py1-1_ovito-compute-rdf_v3.py`
- 绘图rdf: `260410-1u1_ovito-compute-plot_cn-rdf-adf_fr260410-1/py2-1_plt-rdf_v3.py`
- 计算bld, adf: `260410-1u1_ovito-compute-plot_cn-rdf-adf_fr260410-1/py1-3_ovito-compute-bld-adf_v3.py `
- 绘图bld, adf: `260410-1u1_ovito-compute-plot_cn-rdf-adf_fr260410-1/py2-3_plt-bld-adf_v3.py`
- 计算cn: `260410-1u1_ovito-compute-plot_cn-rdf-adf_fr260410-1/py1-2_ovito-compute-cn_v3.py`
- 绘图cn: `260410-1u1_ovito-compute-plot_cn-rdf-adf_fr260410-1/py2-2_bar-plt-cn_v3.py`


# I. exe: `VASP`/`GPUMD` computes EOS

- Workflow: 1) `VASP`优化晶胞/原包 --> 2) 使用脚本程序格式化优化后的晶胞/原包得到`conv.vasp` --> 3) 使用脚本程序生成三轴同比应变结构 --> 4) `VASP`/`NEP-GPUMD`单点计算 --> 5) 表格化数据并绘图
1. `jobid=260224-1_IBRION1-optimize-Li`, `IBRION=2/1`优化原包或晶胞 
2. `jobid=260224-1u2_identify-prim-conv-orth_Li_fr260224-1`, 使用`py_identify-vasp-stru_2_prim-conv-orth.py` identify 优化后的`CONTCAR`, 以
    获取对应的`prim.vasp`, `conv.vasp`, `orth.vasp`等
3. `jobid=260227-1u1_create-triaxial-strain-Li-struc-for-EOS_fr260224-1u2`, `1py1-1_gen_isotropic_strain_conv.py`读取`conv.vasp` 生成三轴同比应变结构
4. `jobid=260227-1_IBRION-1-scf_triaxial-EOS_Li_fr260227-1u1`, VASP单点计算； `jobid=260227-2_gpumd-single-point-triaxial-EOS_Li_fr260227-1u1`, GPUMD单点计算
5. `jobid=260227-3u2_table-plot-triaxial-EOS_Li_DFT-NEP_fr260227-1-260227-2`, 表格化DFT和NEP数据，并绘图





# I. exe: `VASP`/`GPUMD` computes uniaxial deformation: potential energy vs. strain

- Workflow: 1) `VASP`优化晶胞/原包 --> 2) 使用脚本程序格式化优化后的晶胞/原包得到`orth.vasp` --> 3) 使用脚本程序对称性分析并遍历非等价轴,生成单轴应变结构 --> 4) `VASP`/`NEP-GPUMD`单点计算 --> 5) 表格化数据并绘图
1. `jobid=260224-1_IBRION1-optimize-Li`, `IBRION=2/1`优化原包或晶胞
2. `jobid=260224-1u2_identify-prim-conv-orth_Li_fr260224-1`, 使用`py_identify-vasp-stru_2_prim-conv-orth.py` identify 优化后的`CONTCAR`, 以
    获取对应的`prim.vasp`, `conv.vasp`, `orth.vasp`等
3. `jobid=260229-1u1_create-uniaxial-strain-Li-struc_fr260224-1u2`, `1py1-1_gen_uniaxial_strain_orth.py`读取`orth.vasp`生成所有非等价轴的系列单轴应变结构
4. `jobid=260229-1_IBRION-1-scf_uniaxial_Li_fr260229-1u1`, VASP单点计算；`jobid=260229-2_gpumd-single-point_uniaxial_Li_fr260229-1u1`, GPUMD单点计算
5. `jobid=260229-3u1_table-plot-Li-uniaxial-energy-vs-strain_DFT-NEP_fr260229-1-260229-2`, 表格化DFT和NEP数据，并绘图








# I. `VASP-AIMD` (待处理)

## II. NVE AIMD

- `vaspJob_aimd-nve_mdalgo0-smass-3`: NVE

## II. NVT AIMD

- `vaspJob_aimd-nvt_mdalgo0-smass0_DEPRECATED`: MDALGO=0, SMASS=0是vasp早期的nose-hoover thermostat的NVT实现
- `vaspJob_aimd-nvt_mdalgo2-smass0`: 后期发展的nose-hoover thermostat的NVT

> 二者的结果是相同的。早期版本的缺陷是轨迹XDATCAR的坐标被wraped into盒子内(分数坐标始终小于1），并且现在已经放弃维护了。因此不赞成使用了；
> 而后来的版本避免了坐标wrap，分数坐标可以大于1，这就可以直接基于轨迹文件来计算MSD或diffusion性质了。

## II. velocity-rescaling MD

- `vaspJob_aimd-vel-rescal_mdalgo0-smass-1-nblock1`: MDALGO=0和SMASS=-1同时存在时，二者是两种温度控制算法的一对矛盾。由于SMASS=-1的优先级高于IBRION=0, 所以体系选择SMASS=-1的紧致的速度标度的温控算法。也就是说，NBLOCK=1时，AIMD的温度始终被严格缩放T=TBEG + (TEND-TBEG)\*NSTEP/NSW. 
- `vaspJob_aimd-vel-rescal_mdalgo0-smass-1-nblock10`: NBLOCK=10,意味着第NSTEP=1，11，21 ...，步时采用了紧致的速度标度。之间的AIMD是微正则NVE系综模拟。
- `vaspJob_aimd-vel-rescal_mdalgo2-smass-1_ILL`: 病态的速度标度MD, MDALGO=2优先级较高，意味着使用nose-hoover thermstat算法， SMASS=-1又意味着速度标度算法。二者冲突，但由于MDALGO=2的优先级较高而导致SMASS=-1失效。所以模拟的结果，速度标失效。没有温度耦合，所以本质上是把体系近似看成了孤立体系的NVE模拟。

> `vaspJob_aimd-vel-rescal_mdalgo0-smass-1-nblock1`和`vaspJob_aimd-vel-rescal_mdalgo0-smass-1-nblock10`是正确的速度标度MD；而`vaspJob_aimd-vel-rescal_mdalgo2-smass-1_ILL`本质是NVE，是错误的速度标度
