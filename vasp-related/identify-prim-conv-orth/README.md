py_identify-vasp-stru_2_prim-conv-orth.py --name CONTCAR ./   : 把当前路径下的名字为CONTCAR的vasp结构文件做递归遍历，并把转化的conv.vasp, orth.vasp, prim.vasp和对应的CONTCAR存放在相同的目录下。

py_identify-vasp-stru_2_prim-conv-orth.py --name CONTCAR 是 py_identify-vasp-stru_2_prim-conv-orth.py --name CONTCAR ./的一种缺省，二者执行效果等同

py_identify-vasp-stru_2_prim-conv-orth.py --name CONTCAR read_dir  本质上和上同

 py_identify-vasp-stru_2_prim-conv-orth.py --name CONTCAR read_dir write_dir  : 从read_dir目录下遍历, 并把CONTCAR, conv.vasp, orth.vasp, prim.vasp四个文件一起存放在write_dir相应的同级目录里。如果read_dir下有A,B,C等子目录，write_dir下也会自动生成对等的子目录。
