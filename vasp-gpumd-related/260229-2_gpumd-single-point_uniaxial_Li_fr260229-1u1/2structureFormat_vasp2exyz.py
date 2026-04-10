#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
from ase.io import read, write

def main():
    """
    Convert VASP structure files (POSCAR / *.vasp) to extended XYZ,
    and build a supercell by repeating (Nx, Ny, Nz).

    Usage:
      ./structureFormat_vasp2exyz.py
      ./structureFormat_vasp2exyz.py ./
      ./structureFormat_vasp2exyz.py path/to/dir
      ./structureFormat_vasp2exyz.py path/to/dir Nx Ny Nz

    Notes:
      - If Nx, Ny, Nz are not given, default to (4, 5, 6).
      - Recursively search for:
          **/POSCAR
          **/*.vasp
      - Output file is written to the same directory as the input file:
          POSCAR  -> POSCAR_N{Nx}x{Ny}x{Nz}.xyz
          xxx.vasp -> xxx_N{Nx}x{Ny}x{Nz}.xyz
    """
    # -------- parse args --------
    search_dir = "."
    Nx, Ny, Nz = 10, 10, 10

    if len(sys.argv) >= 2:
        search_dir = sys.argv[1]

    if len(sys.argv) == 5:
        try:
            Nx = int(sys.argv[2])
            Ny = int(sys.argv[3])
            Nz = int(sys.argv[4])
        except ValueError:
            print("错误：Nx Ny Nz 必须是整数。例如：4 5 6")
            sys.exit(2)
    elif len(sys.argv) not in (1, 2, 5):
        print("用法错误。\n"
              "用法示例：\n"
              "  ./structureFormat_vasp2exyz.py\n"
              "  ./structureFormat_vasp2exyz.py ./\n"
              "  ./structureFormat_vasp2exyz.py xxx/yyy\n"
              "  ./structureFormat_vasp2exyz.py xxx/yyy 4 5 6\n")
        sys.exit(2)

    if Nx <= 0 or Ny <= 0 or Nz <= 0:
        print("错误：Nx Ny Nz 必须为正整数。")
        sys.exit(2)

    if not os.path.isdir(search_dir):
        print(f"错误：目录不存在：{search_dir}")
        sys.exit(1)

    # -------- recursively find files --------
    # Make search_dir absolute for safety
    root = os.path.abspath(search_dir)

    poscar_files = sorted(glob.glob(os.path.join(root, "**", "POSCAR"), recursive=True))
    vasp_files = sorted(glob.glob(os.path.join(root, "**", "*.vasp"), recursive=True))

    files = poscar_files + vasp_files
    if not files:
        print(f"未在目录 {search_dir}（递归）下发现 POSCAR 或 *.vasp 文件。")
        sys.exit(1)

    print(f"搜索根目录：{root}")
    print(f"超胞倍数  ：Nx={Nx}, Ny={Ny}, Nz={Nz}")
    print(f"发现 {len(files)} 个结构文件（递归），开始转换与生成超胞...\n")

    ok, fail = 0, 0
    for i, in_path in enumerate(files, start=1):
        in_dir = os.path.dirname(in_path)
        base = os.path.basename(in_path)

        # output name rule
        if base == "POSCAR":
            out_base = "POSCAR"
        else:
            out_base, _ = os.path.splitext(base)  # xxx.vasp -> xxx

        out_name = f"{out_base}_{Nx}x{Ny}x{Nz}.xyz"
        out_path = os.path.join(in_dir, out_name)

        try:
            print(f"[{i:04d}/{len(files)}] 读取：{in_path}")
            atoms = read(in_path, format="vasp")

            atoms_sc = atoms.repeat((Nx, Ny, Nz))

            # write extended xyz
            write(out_path, atoms_sc, format="extxyz")

            print(f"               已写出：{out_path}\n")
            ok += 1
        except Exception as e:
            print(f"               ✖ 转换失败：{in_path}\n               原因：{e}\n")
            fail += 1

    print(f"转换完成：成功 {ok} 个，失败 {fail} 个。")


if __name__ == "__main__":
    main()

