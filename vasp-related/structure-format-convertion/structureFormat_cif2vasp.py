#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
from ase.io import read, write

def main():
    cif_files = sorted(glob.glob("*.cif"))
    if not cif_files:
        print("未在当前目录发现 *.cif 文件。")
        sys.exit(1)

    print(f"发现 {len(cif_files)} 个 CIF 文件，开始转换...\n")

    ok, fail = 0, 0
    for i, cif_path in enumerate(cif_files, start=1):
        base, _ = os.path.splitext(cif_path)
        out_path = f"{base}.vasp"
        try:
            print(f"[{i:02d}/{len(cif_files)}] 读取：{cif_path}")
            atoms = read(cif_path)

            # 写出为 VASP POSCAR 格式：
            # - format='vasp'：指定 VASP 写出器
            # - vasp5=True：在 POSCAR 第6行写出元素符号（VASP5 风格）
            # - direct=True：使用分数坐标（Direct）
            # - sort=False：不按元素重排，尽量保持原子顺序
            write(out_path, atoms, format="vasp", vasp5=True, direct=True, sort=False)

            print(f"          已写出：{out_path}\n")
            ok += 1
        except Exception as e:
            print(f"          ✖ 转换失败：{cif_path}\n          原因：{e}\n")
            fail += 1

    print(f"转换完成：成功 {ok} 个，失败 {fail} 个。")

if __name__ == "__main__":
    main()
