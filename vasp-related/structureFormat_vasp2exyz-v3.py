#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
from ase.io import read, write


def is_int_string(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def parse_args():
    """
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
    """
    args = sys.argv[1:]

    src = "."
    dst = None
    Nx, Ny, Nz = 1, 1, 1

    if len(args) == 0:
        # ./script.py
        src = "."
        dst = "."

    elif len(args) == 1:
        # ./script.py path
        src = args[0]
        dst = None  # 后面根据 src 类型决定

    elif len(args) == 3 and all(is_int_string(x) for x in args):
        # 不支持这种写法：./script.py Nx Ny Nz
        print("用法错误：缺少源路径。")
        print_usage_and_exit()

    elif len(args) == 4 and all(is_int_string(x) for x in args[1:4]):
        # ./script.py src Nx Ny Nz
        src = args[0]
        dst = None
        Nx, Ny, Nz = map(int, args[1:4])

    elif len(args) == 2:
        # ./script.py src dst
        src = args[0]
        dst = args[1]

    elif len(args) == 5 and all(is_int_string(x) for x in args[2:5]):
        # ./script.py src dst Nx Ny Nz
        src = args[0]
        dst = args[1]
        Nx, Ny, Nz = map(int, args[2:5])

    else:
        print("用法错误。")
        print_usage_and_exit()

    if Nx <= 0 or Ny <= 0 or Nz <= 0:
        print("错误：Nx Ny Nz 必须为正整数。")
        sys.exit(2)

    return src, dst, Nx, Ny, Nz


def print_usage_and_exit():
    print(
        "用法示例：\n"
        "  ./structureFormat_vasp2exyz.py\n"
        "  ./structureFormat_vasp2exyz.py ./\n"
        "  ./structureFormat_vasp2exyz.py path/to/dir\n"
        "  ./structureFormat_vasp2exyz.py path/to/file\n"
        "  ./structureFormat_vasp2exyz.py path/to/dir 4 5 6\n"
        "  ./structureFormat_vasp2exyz.py path/to/file 4 5 6\n"
        "  ./structureFormat_vasp2exyz.py path/to/src path/to/dst\n"
        "  ./structureFormat_vasp2exyz.py path/to/src path/to/dst 4 5 6\n"
    )
    sys.exit(2)


def collect_files(src_path):
    """
    如果 src_path 是目录：
      递归查找 **/POSCAR, **/CONTCAR, **/*.vasp
    如果 src_path 是文件：
      只处理该文件（无论名字是否为 POSCAR/CONTCAR/*.vasp，均尝试按 VASP 格式读取）
    """
    if os.path.isfile(src_path):
        return [os.path.abspath(src_path)], "file"

    if os.path.isdir(src_path):
        root = os.path.abspath(src_path)
        poscar_files = glob.glob(os.path.join(root, "**", "POSCAR"), recursive=True)
        contcar_files = glob.glob(os.path.join(root, "**", "CONTCAR"), recursive=True)
        vasp_files = glob.glob(os.path.join(root, "**", "*.vasp"), recursive=True)

        files = sorted(set(os.path.abspath(x) for x in (poscar_files + contcar_files + vasp_files)))
        return files, "dir"

    print(f"错误：路径不存在：{src_path}")
    sys.exit(1)


def get_output_basename(in_path):
    """
    输出文件名规则：
    - POSCAR   -> POSCAR
    - CONTCAR  -> CONTCAR
    - xxx.vasp -> xxx
    - 其他任意单文件名 -> 原文件名
    """
    base = os.path.basename(in_path)

    if base in ("POSCAR", "CONTCAR"):
        return base
    elif base.lower().endswith(".vasp"):
        return os.path.splitext(base)[0]
    else:
        return base


def build_output_path(in_path, src_root, dst_root, src_type, Nx, Ny, Nz):
    """
    构造输出路径：
    - 若不超胞（1x1x1），输出名不带 _Nx xNy xNz
    - 若超胞，则带后缀 _NxxNyxNz
    - 若 src 是目录 且 dst_root 与 src_root 不同，则保留相对子目录结构
    - 若 src 是单文件：
        输出直接放到 dst_root 下
    """
    out_base = get_output_basename(in_path)

    if (Nx, Ny, Nz) == (1, 1, 1):
        out_name = f"{out_base}.xyz"
    else:
        out_name = f"{out_base}_{Nx}x{Ny}x{Nz}.xyz"

    if src_type == "file":
        out_dir = dst_root
    else:
        rel_dir = os.path.relpath(os.path.dirname(in_path), src_root)
        if rel_dir == ".":
            out_dir = dst_root
        else:
            out_dir = os.path.join(dst_root, rel_dir)

    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, out_name)


def main():
    src, dst, Nx, Ny, Nz = parse_args()

    src_abs = os.path.abspath(src)
    files, src_type = collect_files(src_abs)

    if not files:
        if src_type == "dir":
            print(f"未在目录 {src_abs}（递归）下发现 POSCAR、CONTCAR 或 *.vasp 文件。")
        else:
            print(f"未找到可处理的文件：{src_abs}")
        sys.exit(1)

    # 决定默认输出位置
    if dst is None:
        if src_type == "file":
            dst_abs = os.path.dirname(src_abs)
        else:
            dst_abs = src_abs
    else:
        dst_abs = os.path.abspath(dst)
        os.makedirs(dst_abs, exist_ok=True)

    print(f"输入路径类型：{src_type}")
    print(f"输入路径    ：{src_abs}")
    print(f"输出根目录  ：{dst_abs}")
    print(f"超胞倍数    ：Nx={Nx}, Ny={Ny}, Nz={Nz}")
    print(f"发现 {len(files)} 个结构文件，开始转换...\n")

    ok, fail = 0, 0

    for i, in_path in enumerate(files, start=1):
        try:
            out_path = build_output_path(
                in_path=in_path,
                src_root=src_abs if src_type == "dir" else os.path.dirname(src_abs),
                dst_root=dst_abs,
                src_type=src_type,
                Nx=Nx,
                Ny=Ny,
                Nz=Nz
            )

            print(f"[{i:04d}/{len(files)}] 读取：{in_path}")
            atoms = read(in_path, format="vasp")

            if (Nx, Ny, Nz) == (1, 1, 1):
                atoms_out = atoms
            else:
                atoms_out = atoms.repeat((Nx, Ny, Nz))

            write(out_path, atoms_out, format="extxyz")

            print(f"               已写出：{out_path}\n")
            ok += 1

        except Exception as e:
            print(f"               ✖ 转换失败：{in_path}\n               原因：{e}\n")
            fail += 1

    print(f"转换完成：成功 {ok} 个，失败 {fail} 个。")


if __name__ == "__main__":
    main()
    
