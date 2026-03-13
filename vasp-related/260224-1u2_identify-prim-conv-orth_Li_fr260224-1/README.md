- `py_identify-vasp-stru_2_prim-conv-orth.py --name CONTCAR ./`   : 递归遍历名为“CONTCAR”的结构，并把转化的“conv.vasp”, “orth.vasp”, “prim.vasp”放在和对应CONTCAR相同的目录下。

- `py_identify-vasp-stru_2_prim-conv-orth.py --name CONTCAR` 是 `py_identify-vasp-stru_2_prim-conv-orth.py --name CONTCAR ./`的一种缺省. 二者执行效果等同

- `py_identify-vasp-stru_2_prim-conv-orth.py --name CONTCAR read_dir`  本质上和上同

- `py_identify-vasp-stru_2_prim-conv-orth.py --name CONTCAR read_dir write_dir`  : 从`read_dir`目录下遍历, 并把"CONTCAR", "conv.vasp", "orth.vasp", "prim.vasp"四个文件一起存放在"write_dir"相应的同级目录里。如果"read_dir"下有"A","B","C"等子目录，"write_dir"下也会自动生成对等的子目录。
