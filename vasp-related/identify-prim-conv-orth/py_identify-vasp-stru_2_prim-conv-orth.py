#!/usr/bin/env python3
"""
Convert VASP structures (POSCAR/CONTCAR/*.vasp) to standardized
primitive, conventional, and orthorhombic cells using ASE + spglib.

usage: ./py_identify-vasp-stru_2_prim-conv-orth.py dir/CONTCAR

Outputs go into the selected directory:
  prim.vasp
  conv.vasp
  orth.vasp (only for hex/trigonal, or already orthogonal)
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
    return (
        atoms.cell.array,
        atoms.get_scaled_positions(),
        atoms.get_atomic_numbers(),
    )


def spglib_to_atoms(cell_tuple):
    if cell_tuple is None:
        return None
    lattice, frac, nums = cell_tuple
    return Atoms(
        numbers=np.asarray(nums, int),
        scaled_positions=np.asarray(frac, float),
        cell=np.asarray(lattice, float),
        pbc=True,
    )


def standardized_cell(atoms, to_primitive, symprec, angle_tol):
    std = spglib.standardize_cell(
        atoms_to_spglib_cell(atoms),
        to_primitive=to_primitive,
        no_idealize=False,
        symprec=symprec,
        angle_tolerance=angle_tol,
    )
    return spglib_to_atoms(std)


def get_spacegroup_info(atoms, symprec, angle_tol):
    ds = spglib.get_symmetry_dataset(
        atoms_to_spglib_cell(atoms),
        symprec=symprec,
        angle_tolerance=angle_tol,
    )
    if ds is None:
        return "P1", 1
    return ds["international"], ds["number"]


def species_order_like(reference, target):
    order = []
    seen = set()
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
    for idxs in sym2idx.values():
        new_idx.extend(idxs)

    return target[new_idx]


def looks_primitive(original, prim):
    if len(original) != len(prim):
        return False

    def params(a):
        return np.r_[a.cell.lengths(), a.cell.angles()]

    return np.allclose(params(original), params(prim), atol=1e-3, rtol=5e-4)


def ensure_right_handed(atoms, eps=1e-12):
    a = atoms.copy()
    cell = a.cell.array.copy()
    if np.linalg.det(cell) < -eps:
        # swap a and b
        cell = cell[[1, 0, 2], :]
        a.set_cell(cell, scale_atoms=False)
        sp = a.get_scaled_positions()
        a.set_scaled_positions(sp[:, [1, 0, 2]])

        if np.linalg.det(a.cell.array) < -eps:
            # fallback: swap a and c
            cell2 = a.cell.array.copy()[[2, 1, 0], :]
            a.set_cell(cell2, scale_atoms=False)
            sp2 = a.get_scaled_positions()
            a.set_scaled_positions(sp2[:, [2, 1, 0]])
    return a


def safe_write_vasp(path, atoms, direct=True, vasp5=True, sort=True):
    try:
        write(path, atoms, format="vasp", direct=direct, vasp5=vasp5, sort=sort)
    except TypeError:
        write(path, atoms, format="vasp", direct=direct)


def is_orthogonal_cell(atoms, ang_tol=1e-2):
    a, b, c = atoms.cell.angles()
    return (abs(a - 90) < ang_tol) and (abs(b - 90) < ang_tol) and (abs(c - 90) < ang_tol)


def is_hex_like(atoms, len_rtol=3e-3, ang_tol=1e-2):
    a, b, c = atoms.cell.lengths()
    alpha, beta, gamma = atoms.cell.angles()
    ab_close = abs(a - b) <= len_rtol * max(a, b)
    hex_angles = (
        abs(alpha - 90) < ang_tol
        and abs(beta - 90) < ang_tol
        and abs(gamma - 120) < ang_tol
    )
    return ab_close and hex_angles


def build_hex_to_orthorhombic(conv):
    S = np.array([[1, 1, 0], [1, -1, 0], [0, 0, 1]], int)
    return make_supercell(conv, S)


def try_make_orthorhombic(conv):
    if is_orthogonal_cell(conv):
        return conv.copy()
    if is_hex_like(conv):
        return build_hex_to_orthorhombic(conv)
    return None


def main():
    ap = argparse.ArgumentParser(description="Standardize VASP structures into prim/conv/orth cells.")
    ap.add_argument("inputs", nargs="+", help="Input POSCAR/CONTCAR/*.vasp files.")
    ap.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Optional output directory (default: current directory).",
    )
    ap.add_argument("--symprec", type=float, default=1e-3)
    ap.add_argument("--angle-tol", type=float, default=5.0)
    ap.add_argument("--preserve-order", action="store_true")
    args = ap.parse_args()

    # Determine output directory
    if args.output_dir is None:
        out_dir = os.getcwd()
    else:
        out_dir = os.path.abspath(os.path.expanduser(args.output_dir))
        os.makedirs(out_dir, exist_ok=True)

    print(f"Output directory: {out_dir}")

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
            print(f"[SKIP] {path}: cannot read ({e})")
            continue

        # Standardize
        try:
            prim = standardized_cell(atoms_in, True, args.symprec, args.angle_tol)
            conv = standardized_cell(atoms_in, False, args.symprec, args.angle_tol)
        except Exception as e:
            print(f"[SKIP] {path}: spglib failed ({e})")
            continue

        if prim is None or conv is None:
            print(f"[SKIP] {path}: spglib returned None")
            continue

        if args.preserve_order:
            prim = species_order_like(atoms_in, prim)
            conv = species_order_like(atoms_in, conv)

        # Ensure right-handed
        prim = ensure_right_handed(prim)
        conv = ensure_right_handed(conv)

        sg_in, num_in = get_spacegroup_info(atoms_in, args.symprec, args.angle_tol)
        sg_prim, num_prim = get_spacegroup_info(prim, args.symprec, args.angle_tol)
        sg_conv, num_conv = get_spacegroup_info(conv, args.symprec, args.angle_tol)

        already_prim = looks_primitive(atoms_in, prim)

        # Output paths
        out_prim = os.path.join(out_dir, "prim.vasp")
        out_conv = os.path.join(out_dir, "conv.vasp")
        out_orth = os.path.join(out_dir, "orth.vasp")

        sort_flag = not args.preserve_order
        safe_write_vasp(out_prim, prim, vasp5=True, sort=sort_flag)
        safe_write_vasp(out_conv, conv, vasp5=True, sort=sort_flag)

        # Orthorhombic
        orth = try_make_orthorhombic(conv)
        if orth is not None:
            orth = ensure_right_handed(orth)
            safe_write_vasp(out_orth, orth, vasp5=True, sort=sort_flag)
            print(f"  Orthorhombic generated.")
        else:
            print("  Orthorhombic not applicable.")

        # Print summary
        def det_str(a):
            return f"{np.linalg.det(a.cell.array): .6f}"

        print(f"[OK] {path}")
        print(f"  Input        : SG {sg_in} (#{num_in}), atoms={len(atoms_in)}, det={det_str(atoms_in)}")
        print(f"  Primitive    : SG {sg_prim} (#{num_prim}), atoms={len(prim)}, det={det_str(prim)}"
              f"{' [input already primitive]' if already_prim else ''}")
        print(f"  Conventional : SG {sg_conv} (#{num_conv}), atoms={len(conv)}, det={det_str(conv)}")
        if orth is not None:
            print(f"  Orthorhombic : atoms={len(orth)}, det={det_str(orth)}")


if __name__ == "__main__":
    main()
