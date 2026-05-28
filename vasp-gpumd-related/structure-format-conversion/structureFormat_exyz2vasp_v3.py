#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
from ase.io import read
from ase.io.vasp import write_vasp


def parse_args():
    """
    支持的用法：

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
    """
    args = sys.argv[1:]

    src = "."
    dst = None

    if len(args) == 0:
        src = "."
        dst = "."

    elif len(args) == 1:
        src = args[0]
        dst = None

    elif len(args) == 2:
        src = args[0]
        dst = args[1]

    else:
        print("用法错误。")
        print_usage_and_exit()

    return src, dst


def print_usage_and_exit():
    print(
        "用法示例：\n"
        "  ./structureFormat_exyz2vasp.py\n"
        "  ./structureFormat_exyz2vasp.py ./\n"
        "  ./structureFormat_exyz2vasp.py path/to/dir\n"
        "  ./structureFormat_exyz2vasp.py path/to/file.xyz\n"
        "  ./structureFormat_exyz2vasp.py path/to/src path/to/dst\n"
    )
    sys.exit(2)


def collect_files(src_path):
    """
    如果 src_path 是目录：
      递归查找 **/*.xyz, **/*.exyz, **/*.extxyz
    如果 src_path 是文件：
      只处理该文件
    """
    if os.path.isfile(src_path):
        return [os.path.abspath(src_path)], "file"

    if os.path.isdir(src_path):
        root = os.path.abspath(src_path)

        xyz_files = glob.glob(os.path.join(root, "**", "*.xyz"), recursive=True)
        exyz_files = glob.glob(os.path.join(root, "**", "*.exyz"), recursive=True)
        extxyz_files = glob.glob(os.path.join(root, "**", "*.extxyz"), recursive=True)

        files = sorted(set(os.path.abspath(x) for x in (xyz_files + exyz_files + extxyz_files)))
        return files, "dir"

    print(f"错误：路径不存在：{src_path}")
    sys.exit(1)


def get_output_basename(in_path):
    """
    输出文件名基础名：
    - xxx.xyz    -> xxx
    - xxx.exyz   -> xxx
    - xxx.extxyz -> xxx
    - 其他文件名  -> 原文件名去后缀
    """
    base = os.path.basename(in_path)
    stem, _ = os.path.splitext(base)
    return stem


def build_output_dir(in_path, src_root, dst_root, src_type):
    """
    构造输出目录：
    - 若 src 是目录 且 dst_root 与 src_root 不同，则保留相对子目录结构
    - 若 src 是单文件，输出直接放到 dst_root 下
    """
    if src_type == "file":
        out_dir = dst_root
    else:
        rel_dir = os.path.relpath(os.path.dirname(in_path), src_root)
        if rel_dir == ".":
            out_dir = dst_root
        else:
            out_dir = os.path.join(dst_root, rel_dir)

    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def build_output_path(out_dir, out_base, iframe, nframes):
    """
    构造每一帧对应的 vasp 文件名。

    如果只有一帧：
        xxx.vasp
    如果有多帧：
        xxx_frame-000001.vasp
        xxx_frame-000002.vasp
        ...
    """
    if nframes == 1:
        out_name = f"{out_base}.vasp"
    else:
        out_name = f"{out_base}_frame-{iframe+1:06d}.vasp"

    return os.path.join(out_dir, out_name)


def main():
    src, dst = parse_args()

    src_abs = os.path.abspath(src)
    files, src_type = collect_files(src_abs)

    if not files:
        if src_type == "dir":
            print(f"未在目录 {src_abs}（递归）下发现 *.xyz、*.exyz 或 *.extxyz 文件。")
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
    print(f"发现 {len(files)} 个 exyz/xyz 文件，开始转换...\n")

    ok_files, fail_files = 0, 0
    ok_frames, fail_frames = 0, 0

    for i, in_path in enumerate(files, start=1):
        try:
            print(f"[{i:04d}/{len(files)}] 读取：{in_path}")

            # ':' 表示读取该 exyz/xyz 文件中的所有帧
            atoms_list = read(in_path, index=":", format="extxyz")

            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]

            nframes = len(atoms_list)

            out_dir = build_output_dir(
                in_path=in_path,
                src_root=src_abs if src_type == "dir" else os.path.dirname(src_abs),
                dst_root=dst_abs,
                src_type=src_type
            )

            out_base = get_output_basename(in_path)

            print(f"               发现 {nframes} 个结构帧")

            for iframe, atoms in enumerate(atoms_list):
                try:
                    out_path = build_output_path(
                        out_dir=out_dir,
                        out_base=out_base,
                        iframe=iframe,
                        nframes=nframes
                    )

                    write_vasp(
                        out_path,
                        atoms,
                        vasp5=True,
                        direct=True,
                        sort=False
                    )

                    print(f"               已写出：{out_path}")
                    ok_frames += 1

                except Exception as e:
                    print(f"               ✖ 第 {iframe+1} 帧转换失败：{e}")
                    fail_frames += 1

            print()
            ok_files += 1

        except Exception as e:
            print(f"               ✖ 文件读取/转换失败：{in_path}\n               原因：{e}\n")
            fail_files += 1

    print("转换完成：")
    print(f"  成功读取文件数：{ok_files}")
    print(f"  失败文件数    ：{fail_files}")
    print(f"  成功写出结构帧：{ok_frames}")
    print(f"  失败结构帧    ：{fail_frames}")


if __name__ == "__main__":
    main()