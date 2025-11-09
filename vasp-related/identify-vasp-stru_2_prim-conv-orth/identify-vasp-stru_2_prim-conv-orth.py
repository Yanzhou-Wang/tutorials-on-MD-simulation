#!/usr/bin/env python3
"""
Convert VASP structures (POSCAR/CONTCAR/*.vasp) to standardized
primitive and conventional cells using ASE + spglib.

Outputs in the same directory:
  prim.vasp  - standardized primitive cell (VASP5 with species + counts)
  conv.vasp  - standardized conventional cell (VASP5 with species + counts)
  orth.vasp  - orthorhombic cell if hex/trigonal (VASP5 with species + counts)
"""

import sys
import os
import glob
import argparse
import numpy as np
from ase import Atoms
from ase.io import read, write
import spglib
from ase.build import make_supercell


def atoms_to_spglib_cell(atoms: Atoms):
    lattice = atoms.cell.array
    frac = atoms.get_scaled_positions()
    nums = atoms.get_atomic_numbers()
    return (lattice, frac, nums)


def spglib_to_atoms(cell_tuple):
    if cell_tuple is None:
        return None
    lattice, frac, nums = cell_tuple
    a = Atoms(
        numbers=np.asarray(nums, dtype=int),
        scaled_positions=np.asarray(frac, dtype=float),
        cell=np.asarray(lattice, dtype=float),
        pbc=True,
    )
    return a


def standardized_cell(atoms: Atoms, to_primitive: bool, symprec: float, angle_tolerance: float):
    std = spglib.standardize_cell(
        atoms_to_spglib_cell(atoms),
        to_primitive=to_primitive,
        no_idealize=False,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    return spglib_to_atoms(std)


def get_spacegroup_info(atoms: Atoms, symprec: float, angle_tolerance: float):
    ds = spglib.get_symmetry_dataset(atoms_to_spglib_cell(atoms), symprec=symprec, angle_tolerance=angle_tolerance)
    if ds is None:
        return "P1", 1
    return ds["international"], ds["number"]


def species_order_like(reference: Atoms, target: Atoms) -> Atoms:
    order, seen = [], set()
    for s in reference.get_chemical_symbols():
        if s not in seen:
            order.append(s)
            seen.add(s)
    sym2idx = {}
    for idx, s in enumerate(target.get_chemical_symbols()):
        sym2idx.setdefault(s, []).append(idx)
    new_idx = []
    for s in order:
        new_idx.extend(sym2idx.pop(s, []))
    for _, idxs in sym2idx.items():
        new_idx.extend(idxs)
    return target[new_idx]


def looks_primitive(original: Atoms, prim: Atoms) -> bool:
    if len(original) != len(prim):
        return False
    def params(a: Atoms):
        return np.r_[a.cell.lengths(), a.cell.angles()]
    return np.allclose(params(original), params(prim), atol=1e-3, rtol=5e-4)


# === FIX: 保证右手系（det(cell) > 0）；通过交换 a,b 并同步交换分数坐标列来实现 ===
def ensure_right_handed(atoms: Atoms, eps: float = 1e-12) -> Atoms:
    a = atoms.copy()
    cell = a.cell.array.copy()  # shape (3,3)
    det = np.linalg.det(cell)
    if det < -eps:
        # swap a and b
        cell = cell[[1, 0, 2], :]
        a.set_cell(cell, scale_atoms=False)
        sp = a.get_scaled_positions()
        sp = sp[:, [1, 0, 2]]  # swap fractional columns x <-> y
        a.set_scaled_positions(sp)

        # 再验证一次
        det2 = np.linalg.det(a.cell.array)
        if det2 < -eps:
            # 极少数情况下（数值病态），再做一次 a<->c 兜底
            cell2 = a.cell.array.copy()
            cell2 = cell2[[2, 1, 0], :]
            a.set_cell(cell2, scale_atoms=False)
            sp2 = a.get_scaled_positions()
            sp2 = sp2[:, [2, 1, 0]]
            a.set_scaled_positions(sp2)
    return a


# === FIX: 总是以 VASP5 写出（显式元素行 + 计数行）；保留回退以兼容旧 ase ===
def safe_write_vasp(path, atoms, direct=True, vasp5=True, sort=True):
    try:
        write(path, atoms, format="vasp", direct=direct, vasp5=vasp5, sort=sort)
    except TypeError:
        # 部分旧版 ASE 可能不接受 vasp5/sort 关键字
        write(path, atoms, format="vasp", direct=direct)


def is_orthogonal_cell(atoms, ang_tol=1e-2):
    a, b, c = atoms.cell.angles()
    return (abs(a - 90) < ang_tol) and (abs(b - 90) < ang_tol) and (abs(c - 90) < ang_tol)


def is_hex_like(atoms, len_rtol=3e-3, ang_tol=1e-2):
    (a, b, c) = atoms.cell.lengths()
    (alpha, beta, gamma) = atoms.cell.angles()
    ab_close = abs(a - b) <= len_rtol * max(a, b)
    angles_hex = (abs(alpha - 90) < ang_tol) and (abs(beta - 90) < ang_tol) and (abs(gamma - 120) < ang_tol)
    return ab_close and angles_hex


def build_hex_to_orthorhombic(conv):
    S = np.array([[1,  1, 0],
                  [1, -1, 0],
                  [0,  0, 1]], dtype=int)
    ortho = make_supercell(conv, S)
    return ortho


def try_make_orthorhombic(conv, sg_info=None):
    if is_orthogonal_cell(conv):
        return conv.copy()
    if is_hex_like(conv):
        return build_hex_to_orthorhombic(conv)
    return None


def main():
    ap = argparse.ArgumentParser(description="Convert VASP structures to standardized primitive and conventional cells.")
    ap.add_argument("inputs", nargs="+", help="POSCAR/CONTCAR/*.vasp files (glob allowed).")
    ap.add_argument("--symprec", type=float, default=1e-3, help="Distance tolerance for spglib symmetry detection.")
    ap.add_argument("--angle-tol", type=float, default=5.0, help="Angle tolerance (degrees) for spglib.")
    ap.add_argument("--preserve-order", action="store_true",
                    help="Reorder output atoms so species ordering follows the input file (useful for POTCAR).")
    ap.add_argument("--vasp5", action="store_true", help="(kept for compatibility) Write VASP5 style.")
    args = ap.parse_args()

    files = []
    for patt in args.inputs:
        files.extend(glob.glob(patt))
    if not files:
        print("No input files matched.")
        sys.exit(1)

    for path in files:
        try:
            atoms_in = read(path, format="vasp")
        except Exception as e:
            print(f"[SKIP] {path}: failed to read ({e})")
            continue

        try:
            prim = standardized_cell(atoms_in, to_primitive=True,
                                     symprec=args.symprec, angle_tolerance=args.angle_tol)
            conv = standardized_cell(atoms_in, to_primitive=False,
                                     symprec=args.symprec, angle_tolerance=args.angle_tol)
        except Exception as e:
            print(f"[SKIP] {path}: spglib standardization failed ({e})")
            continue

        if prim is None or conv is None:
            print(f"[SKIP] {path}: spglib returned None (try larger --symprec, e.g., 1e-2)")
            continue

        if args.preserve_order:
            prim = species_order_like(atoms_in, prim)
            conv = species_order_like(atoms_in, conv)

        # 保证右手系
        prim = ensure_right_handed(prim)
        conv = ensure_right_handed(conv)

        sg_in, n_in = get_spacegroup_info(atoms_in, args.symprec, args.angle_tol)
        sg_prim, n_prim = get_spacegroup_info(prim, args.symprec, args.angle_tol)
        sg_conv, n_conv = get_spacegroup_info(conv, args.symprec, args.angle_tol)
        already_prim = looks_primitive(atoms_in, prim)

        out_dir = os.path.dirname(path)
        out_prim = os.path.join(out_dir, "prim.vasp")
        out_conv = os.path.join(out_dir, "conv.vasp")

        sort_flag = not args.preserve_order
        # 始终 VASP5 写出（元素名 + 计数）
        safe_write_vasp(out_prim, prim, direct=True, vasp5=True, sort=sort_flag)   # NOTE: force vasp5
        safe_write_vasp(out_conv, conv, direct=True, vasp5=True, sort=sort_flag)

        # 生成 orth
        orth = try_make_orthorhombic(conv)
        out_orth = os.path.join(out_dir, "orth.vasp")
        if orth is not None:
            orth = ensure_right_handed(orth)
            safe_write_vasp(out_orth, orth, direct=True, vasp5=True, sort=sort_flag)
            print(f"     Orthorhombic -> orth.vasp : angles={orth.cell.angles()}")
        else:
            print("     Orthorhombic -> N/A (non-hexagonal/non-trigonal systems cannot become strictly orthogonal without strain)")

        # 打印体积符号（右手性确认）
        def det_str(a: Atoms) -> str:
            return f"{np.linalg.det(a.cell.array): .6f}"
        print(f"[OK] {path}")
        print(f"     Input        : SG {sg_in:>4} (#{n_in:>3}), atoms={len(atoms_in)}, det(cell)={det_str(atoms_in)}")
        print(f"     Primitive    -> prim.vasp : SG {sg_prim:>4} (#{n_prim:>3}), atoms={len(prim)}, det(cell)={det_str(prim)}"
              f"{' [input already primitive]' if already_prim else ''}")
        print(f"     Conventional -> conv.vasp : SG {sg_conv:>4} (#{n_conv:>3}), atoms={len(conv)}, det(cell)={det_str(conv)}")
        if orth is not None:
            print(f"     Orthorhombic -> orth.vasp : atoms={len(orth)}, det(cell)={det_str(orth)}")


if __name__ == "__main__":
    main()
