#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a Monkhorst-Pack KPOINTS file from a given KSPACING value.

用法：
    python gen_kpoints_from_kspacing.py 0.2

说明：
  - 假定当前目录下存在 POSCAR（VASP4 / VASP5 均可）
  - 从 POSCAR 读取晶胞 a,b,c (Å)
  - 用 Δk ≈ 2π / (L * N) ≲ KSPACING 估算 (N1, N2, N3)
  - 生成 Gamma-centered 的 Monkhorst-Pack KPOINTS 文件
"""

import sys
import numpy as np
from ase.io import read


def kmesh_from_kspacing(cell_lengths, kspacing, min_n=1, max_n=80):
    """
    根据晶胞长度和 KSPACING 估算 Monkhorst-Pack 网格 (N1, N2, N3).

    cell_lengths: (a, b, c) in Å
    kspacing    : target k-point spacing in Å^-1
    min_n, max_n: N_i 的上下限，防止过小或过大
    """
    a, b, c = cell_lengths
    two_pi = 2.0 * np.pi

    def n_from_L(L):
        if L <= 0:
            return min_n
        n = int(np.ceil(two_pi / (L * kspacing)))
        return max(min_n, min(n, max_n))

    N1 = n_from_L(a)
    N2 = n_from_L(b)
    N3 = n_from_L(c)

    return N1, N2, N3


def write_kpoints(filename, mesh, comment="KPOINTS generated from KSPACING"):
    """
    写出标准 VASP KPOINTS（Γ-centered Monkhorst-Pack）
    """
    N1, N2, N3 = mesh
    with open(filename, "w") as f:
        f.write(f"{comment}\n")
        f.write("0\n")            # 让 VASP 使用下面给出的网格
        f.write("Gamma\n")        # Gamma-centered Monkhorst-Pack
        f.write(f"{N1} {N2} {N3}\n")
        f.write("0 0 0\n")        # shift = Gamma
    print(f"Written KPOINTS with mesh = {N1} {N2} {N3}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python gen_kpoints_from_kspacing.py KSPACING")
        print("Example: python gen_kpoints_from_kspacing.py 0.2")
        sys.exit(1)

    try:
        kspacing = float(sys.argv[1])
    except ValueError:
        print("Error: KSPACING must be a float, e.g. 0.2")
        sys.exit(1)

    poscar = "POSCAR"
    try:
        atoms = read(poscar, format="vasp")
    except Exception as e:
        print(f"Error: failed to read {poscar}: {e}")
        sys.exit(1)

    a, b, c = atoms.cell.lengths()
    print(f"Read POSCAR: a={a:.4f} Å, b={b:.4f} Å, c={c:.4f} Å")
    print(f"Target KSPACING = {kspacing} Å^-1")

    mesh = kmesh_from_kspacing((a, b, c), kspacing)
    print(f"Estimated k-mesh (N1 N2 N3) = {mesh[0]} {mesh[1]} {mesh[2]}")

    write_kpoints("KPOINTS", mesh)


if __name__ == "__main__":
    main()
