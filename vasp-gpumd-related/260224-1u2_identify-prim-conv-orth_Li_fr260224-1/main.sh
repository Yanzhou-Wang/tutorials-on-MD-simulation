#!/bin/bash
./py_identify-vasp-stru_2_prim-conv-orth_v3.py --name "CONTCAR" ../260224-1_* ./

#上述代码执行后直接把read_dir="../260224-1_*"的目录下的CONTCAR以及对应的子目录，和转化好的conv.vasp, orth.vasp, prim.vasp，一并转化到当前目录下了。 执行后的结果，就是继承了read_dir的子目录层级。
