#!/usr/bin/env python3
"""
py_identify-vasp-stru_2_prim-conv-orth.py

Convert VASP structures to standardized primitive and conventional cells using ASE + spglib.

Outputs:
  - prim.vasp  : standardized primitive cell (VASP5 with species + counts)
  - conv.vasp  : standardized conventional cell (VASP5 with species + counts)
  - orth.vasp  : orthorhombic cell if hex/trigonal (VASP5 with species + counts)

Usage (new --name mode, recursive search by exact filename):
  1) Search in current directory (recursive) and write outputs next to each found file to the place where corresponding CONTCAR stay:
       py_identify-vasp-stru_2_prim-conv-orth.py --name CONTCAR
       py_identify-vasp-stru_2_prim-conv-orth.py --name CONTCAR ./

  2) Search in read_dir (recursive) and write outputs next to each found file to the place where corresponding CONTCAR stays:
       py_identify-vasp-stru_2_prim-conv-orth.py --name CONTCAR read_dir

  3) Search in read_dir (recursive), but write outputs into write_dir (keep relative paths),
     and also copy the original input file (NAME) into the mirrored directory:
       py_identify-vasp-stru_2_prim-conv-orth.py --name CONTCAR read_dir write_dir

  NOTE:
    --name can be any filename (e.g., POSCAR / CONTCAR / xxx.vasp), as long as the file
    content is a VASP structure readable by ASE.

Usage (legacy mode, keep compatible with your original script):
  - Provide one or more input files / globs:
       py_identify-vasp-stru_2_prim-conv-orth.py POSCAR CONTCAR *.vasp

Input:
  - VASP structure file(s) readable by ASE (POSCAR/CONTCAR/*.vasp), either via:
      (a) --name NAME (recursive search), or
      (b) legacy positional globs.

Output:
  - In --name mode with one output root (write_dir provided): outputs written under write_dir
    with relative directory structure preserved; also copies the input NAME file.
  - Otherwise: outputs written in the same directory as each input file.
"""

import sys
import os
import glob
import argparse
import shutil
import numpy as np
from pathlib import Path

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


def align_orth_cell_to_axes(atoms: Atoms, eps: float = 1e-10) -> Atoms:
    """
    For an (almost) orthorhombic cell, rotate the whole structure so that
    lattice vectors a,b,c align with x,y,z axes respectively (right-handed),
    and set the cell to diag(|a|,|b|,|c|).
    """
    a0 = atoms.copy()

    cell = a0.cell.array.copy()  # rows: a,b,c in ASE/VASP convention
    a_vec, b_vec, c_vec = cell[0], cell[1], cell[2]

    la = np.linalg.norm(a_vec)
    lb = np.linalg.norm(b_vec)
    lc = np.linalg.norm(c_vec)
    if la < eps or lb < eps or lc < eps:
        return a0

    ua = a_vec / la
    ub = b_vec / lb
    uc = c_vec / lc

    # Check orthogonality; if not close to orthogonal, do nothing
    if (abs(np.dot(ua, ub)) > 1e-6) or (abs(np.dot(ua, uc)) > 1e-6) or (abs(np.dot(ub, uc)) > 1e-6):
        return a0

    # Enforce right-handed orthonormal basis: uc := ua x ub (should be ± original uc)
    uc_rh = np.cross(ua, ub)
    n_uc = np.linalg.norm(uc_rh)
    if n_uc < eps:
        return a0
    uc_rh /= n_uc

    U = np.vstack([ua, ub, uc_rh])  # rows

    pos = a0.get_positions()
    pos_new = pos @ U.T

    cell_new = np.diag([la, lb, lc])

    a0.set_cell(cell_new, scale_atoms=False)
    a0.set_positions(pos_new)
    a0.wrap(eps=1e-12)

    return a0


# === FIX: 总是以 VASP5 写出（显式元素行 + 计数行）；保留回退以兼容旧 ase ===
def safe_write_vasp(path, atoms, direct=True, vasp5=True, sort=True):
    try:
        write(path, atoms, format="vasp", direct=direct, vasp5=vasp5, sort=sort)
    except TypeError:
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


def _process_one_file(path, out_dir, symprec, angle_tol, preserve_order):
    try:
        atoms_in = read(path, format="vasp")
    except Exception as e:
        print(f"[SKIP] {path}: failed to read ({e})")
        return

    try:
        prim = standardized_cell(atoms_in, to_primitive=True,
                                 symprec=symprec, angle_tolerance=angle_tol)
        conv = standardized_cell(atoms_in, to_primitive=False,
                                 symprec=symprec, angle_tolerance=angle_tol)
    except Exception as e:
        print(f"[SKIP] {path}: spglib standardization failed ({e})")
        return

    if prim is None or conv is None:
        print(f"[SKIP] {path}: spglib returned None (try larger --symprec, e.g., 1e-2)")
        return

    if preserve_order:
        prim = species_order_like(atoms_in, prim)
        conv = species_order_like(atoms_in, conv)

    prim = ensure_right_handed(prim)
    conv = ensure_right_handed(conv)

    sg_in, n_in = get_spacegroup_info(atoms_in, symprec, angle_tol)
    sg_prim, n_prim = get_spacegroup_info(prim, symprec, angle_tol)
    sg_conv, n_conv = get_spacegroup_info(conv, symprec, angle_tol)
    already_prim = looks_primitive(atoms_in, prim)

    out_prim = os.path.join(out_dir, "prim.vasp")
    out_conv = os.path.join(out_dir, "conv.vasp")

    sort_flag = not preserve_order
    safe_write_vasp(out_prim, prim, direct=True, vasp5=True, sort=sort_flag)
    safe_write_vasp(out_conv, conv, direct=True, vasp5=True, sort=sort_flag)

    orth = try_make_orthorhombic(conv)
    out_orth = os.path.join(out_dir, "orth.vasp")

    if orth is not None:
        orth = ensure_right_handed(orth)
        orth = align_orth_cell_to_axes(orth)
        safe_write_vasp(out_orth, orth, direct=True, vasp5=True, sort=sort_flag)
        print(f"     Orthorhombic -> orth.vasp : angles={orth.cell.angles()}")
    else:
        print("     Orthorhombic -> N/A (non-hexagonal/non-trigonal systems cannot become strictly orthogonal without strain)")

    def det_str(a: Atoms) -> str:
        return f"{np.linalg.det(a.cell.array): .6f}"

    print(f"[OK] {path}")
    print(f"     Input        : SG {sg_in:>4} (#{n_in:>3}), atoms={len(atoms_in)}, det(cell)={det_str(atoms_in)}")
    print(f"     Primitive    -> prim.vasp : SG {sg_prim:>4} (#{n_prim:>3}), atoms={len(prim)}, det(cell)={det_str(prim)}"
          f"{' [input already primitive]' if already_prim else ''}")
    print(f"     Conventional -> conv.vasp : SG {sg_conv:>4} (#{n_conv:>3}), atoms={len(conv)}, det(cell)={det_str(conv)}")
    if orth is not None:
        print(f"     Orthorhombic -> orth.vasp : atoms={len(orth)}, det(cell)={det_str(orth)}")


def main():
    ap = argparse.ArgumentParser(description="Convert VASP structures to standardized primitive and conventional cells.")
    ap.add_argument("inputs", nargs="*", help="POSCAR/CONTCAR/*.vasp files (glob allowed).")
    ap.add_argument("--name", type=str, default=None,
                    help="Recursively search for files with this exact filename under read_dir (optionally map to write_dir).")
    ap.add_argument("--symprec", type=float, default=1e-3, help="Distance tolerance for spglib symmetry detection.")
    ap.add_argument("--angle-tol", type=float, default=5.0, help="Angle tolerance (degrees) for spglib.")
    ap.add_argument("--preserve-order", action="store_true",
                    help="Reorder output atoms so species ordering follows the input file (useful for POTCAR).")
    ap.add_argument("--vasp5", action="store_true", help="(kept for compatibility) Write VASP5 style.")
    args = ap.parse_args()

    # ------------------------
    # Mode 1: --name recursive mode
    # ------------------------
    if args.name is not None:
        # inputs are treated as [read_dir] or [read_dir, write_dir]
        if len(args.inputs) == 0:
            read_dir = Path(".").resolve()
            write_dir = None
        elif len(args.inputs) == 1:
            read_dir = Path(args.inputs[0]).expanduser().resolve()
            write_dir = None
        elif len(args.inputs) == 2:
            read_dir = Path(args.inputs[0]).expanduser().resolve()
            write_dir = Path(args.inputs[1]).expanduser().resolve()
        else:
            print("ERROR: in --name mode, you can pass 0/1/2 positional args: [read_dir] [write_dir]")
            sys.exit(1)

        if not read_dir.exists() or not read_dir.is_dir():
            print(f"ERROR: read_dir must be an existing directory: {read_dir}")
            sys.exit(1)

        if write_dir is not None:
            if not write_dir.exists() or not write_dir.is_dir():
                print(f"ERROR: write_dir must be an existing directory: {write_dir}")
                sys.exit(1)

        # find files by exact filename
        targets = sorted([p for p in read_dir.rglob(args.name) if p.is_file()])
        if not targets:
            print(f"No file named '{args.name}' found under: {read_dir}")
            sys.exit(1)

        print(f"Search name : {args.name}")
        print(f"Read dir    : {read_dir}")
        print(f"Write dir   : {write_dir if write_dir is not None else '(in-place)'}")
        print(f"Found       : {len(targets)} file(s)\n")

        for path_obj in targets:
            in_path = str(path_obj)

            if write_dir is None:
                out_dir = str(path_obj.parent)  # in-place
            else:
                rel_parent = path_obj.parent.relative_to(read_dir)
                out_parent = write_dir / rel_parent
                out_parent.mkdir(parents=True, exist_ok=True)

                # Copy the original NAME file into mirrored directory
                dst_infile = out_parent / path_obj.name
                try:
                    shutil.copy2(path_obj, dst_infile)
                except Exception as e:
                    print(f"[SKIP] {in_path}: failed to copy to {dst_infile} ({e})")
                    continue

                out_dir = str(out_parent)
                in_path = str(dst_infile)  # process the copied file (ensures self-contained output tree)

            _process_one_file(
                path=in_path,
                out_dir=out_dir,
                symprec=args.symprec,
                angle_tol=args.angle_tol,
                preserve_order=args.preserve_order,
            )

        return

    # ------------------------
    # Mode 2: legacy glob inputs mode (keep your original behavior)
    # ------------------------
    files = []
    for patt in args.inputs:
        files.extend(glob.glob(patt))
    if not files:
        print("No input files matched. (Tip: use --name NAME [read_dir] [write_dir])")
        sys.exit(1)

    for path in files:
        out_dir = os.path.dirname(os.path.abspath(path)) or "."
        _process_one_file(
            path=path,
            out_dir=out_dir,
            symprec=args.symprec,
            angle_tol=args.angle_tol,
            preserve_order=args.preserve_order,
        )


if __name__ == "__main__":
    main()
