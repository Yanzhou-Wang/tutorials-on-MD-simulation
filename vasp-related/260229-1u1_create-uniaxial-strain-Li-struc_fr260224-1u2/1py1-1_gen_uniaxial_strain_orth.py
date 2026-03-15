#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate uniaxial-strained structures from orth.vasp.

Input layout:
  <read_comm_dir>/id-mp-*/orth.vasp

Output layout:
  <out_dir>/id-mp-*/orth_{axis}_{scale:.2f}.vasp
    axis in {a,b,c}, with equivalence dedup:
      - a=b=c  -> only a
      - a=b!=c -> a and c
      - all different -> a,b,c

Rules:
- Must be Direct (fractional) coordinates. If Cartesian -> ERROR and skip this structure.
- Cell vectors must be orthogonal within tol_ortho (dot products <= tol_ortho).
- Apply uniaxial scaling by modifying one lattice vector only.
- Fractional coordinates must remain unchanged (ASE: set_cell(..., scale_atoms=False)).
"""

from __future__ import annotations

from pathlib import Path
import shutil
import numpy as np

from ase.io import read, write

read_comm_dir = Path("../260224-1u2_identify-prim-conv-orth_Li_fr260224-1")
out_dir = Path("./")

in_filename = "orth.vasp"

# strain range: -0.05, -0.04, ..., 0.10  (step=0.01)
strain_start = -0.05
strain_end = 0.10
strain_step = 0.05

# tolerances requested by you
tol_ortho = 1e-3   # for orthogonality check on dot products
tol_len = 1e-3     # for equivalence check on lengths (Angstrom)

def strain_values():
    """Yield strain values with exact 0.01 steps using integer loop."""
    start_i = int(round(strain_start * 100))
    end_i   = int(round(strain_end * 100))
    step_i  = int(round(strain_step * 100))
    for i in range(start_i, end_i + 1, step_i):
        yield i / 100.0

def is_direct_coordinate_file(poscar_path: Path) -> bool:
    """
    Check whether a VASP POSCAR-like file uses Direct (fractional) coordinates.
    Returns True if 'Direct' (or leading 'D') is found as the coord-type line.
    Returns False if 'Cartesian' (or leading 'C') is found.
    If not found, treat as False (unknown -> error).
    """
    txt = poscar_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(txt) < 8:
        return False

    # VASP5: line 6 has element symbols, line 7 has counts
    # Then optional "Selective dynamics" line
    # Next line should be Direct/Cartesian.
    # We'll search in a small window after line 7 for robustness.
    # (0-indexed: 0 comment, 1 scale, 2-4 cell, 5 elems, 6 counts, 7 maybe S/coord)
    start = 7
    end = min(len(txt), 12)
    for i in range(start, end):
        s = txt[i].strip().lower()
        if not s:
            continue
        if s.startswith("s"):  # selective dynamics
            continue
        if s.startswith("d"):  # direct
            return True
        if s.startswith("c"):  # cartesian
            return False
        # If it's something else (e.g., "direct" with spaces is handled),
        # continue searching.
    return False

def check_orthogonal(cell: np.ndarray) -> bool:
    """
    Check if cell vectors are mutually orthogonal using dot products.
    cell: (3,3) rows are a,b,c vectors in Angstrom.
    """
    a, b, c = cell[0], cell[1], cell[2]
    dab = float(np.dot(a, b))
    dac = float(np.dot(a, c))
    dbc = float(np.dot(b, c))
    return (abs(dab) <= tol_ortho) and (abs(dac) <= tol_ortho) and (abs(dbc) <= tol_ortho)

def unique_axes_by_length(lengths: np.ndarray) -> list[int]:
    """
    Decide which axes to apply uniaxial strain to, based on equivalence of lengths.
    lengths: array-like [|a|,|b|,|c|]
    Returns list of axis indices [0,1,2] corresponding to a,b,c after dedup:
      - all equal -> [0]
      - a=b!=c   -> [0,2]
      - a=c!=b   -> [0,1]  (by our rule: keep a, keep b)
      - b=c!=a   -> [0,1]  (keep a, keep b; c is redundant with b)
      - all different -> [0,1,2]
    """
    la, lb, lc = map(float, lengths)

    def eq(x, y) -> bool:
        return abs(x - y) <= tol_len

    ab = eq(la, lb)
    ac = eq(la, lc)
    bc = eq(lb, lc)

    if ab and ac and bc:
        return [0]          # cubic: only a
    if ab and (not ac) and (not bc):
        return [0, 2]       # a=b != c -> a and c
    if ac and (not ab) and (not bc):
        return [0, 1]       # a=c != b -> keep a, keep b
    if bc and (not ab) and (not ac):
        return [0, 1]       # b=c != a -> keep a, keep b
    return [0, 1, 2]        # all different (or near-degenerate but not matching above)

def axis_name(i: int) -> str:
    return ["a", "b", "c"][i]

def main():
    if not read_comm_dir.exists():
        raise FileNotFoundError(f"read_comm_dir not found: {read_comm_dir}")

    # clean output directory
    #if out_dir.exists():
    #    print(f"[WARN] Removing existing output directory: {out_dir}")
    #    shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subdirs = sorted([p for p in read_comm_dir.iterdir() if p.is_dir() and p.name.startswith("id-mp-")])
    print(f"[INFO] Found {len(subdirs)} structures under {read_comm_dir}")
    print(f"[INFO] tol_ortho={tol_ortho:g}, tol_len={tol_len:g}")
    print(f"[INFO] strain range: {strain_start:+.2f} .. {strain_end:+.2f} step {strain_step:.2f}")

    total_written = 0
    total_skipped_missing = 0
    total_skipped_cart = 0
    total_skipped_nonortho = 0

    for idx, sd in enumerate(subdirs, 1):
        in_path = sd / in_filename
        if not in_path.exists():
            print(f"[SKIP] ({idx}/{len(subdirs)}) Missing {in_filename} in {sd.name}")
            total_skipped_missing += 1
            continue

        # Must be Direct (fractional) coordinates
        if not is_direct_coordinate_file(in_path):
            print(f"[ERROR] ({idx}/{len(subdirs)}) {sd.name}: {in_filename} is NOT Direct (fractional). "
                  f"Skip this structure.")
            total_skipped_cart += 1
            continue

        # Read via ASE
        try:
            atoms0 = read(str(in_path), format="vasp")
        except Exception as e:
            print(f"[ERROR] ({idx}/{len(subdirs)}) Failed to read {in_path}: {e}")
            continue

        cell0 = atoms0.get_cell().array  # (3,3)
        if not check_orthogonal(cell0):
            a, b, c = cell0[0], cell0[1], cell0[2]
            dab = float(np.dot(a, b))
            dac = float(np.dot(a, c))
            dbc = float(np.dot(b, c))
            print(f"[WARN]  ({idx}/{len(subdirs)}) {sd.name}: cell not orthogonal "
                  f"(a·b={dab:.3e}, a·c={dac:.3e}, b·c={dbc:.3e}) -> SKIP")
            total_skipped_nonortho += 1
            continue

        lengths = np.array([np.linalg.norm(cell0[0]), np.linalg.norm(cell0[1]), np.linalg.norm(cell0[2])], float)
        axes = unique_axes_by_length(lengths)

        out_subdir = out_dir / sd.name
        out_subdir.mkdir(parents=True, exist_ok=True)

        axes_str = ",".join(axis_name(i) for i in axes)
        print(f"[PROC] ({idx}/{len(subdirs)}) {sd.name} | lengths(a,b,c)={lengths[0]:.6f},"
              f"{lengths[1]:.6f},{lengths[2]:.6f} | axes={axes_str}")

        for ax in axes:
            for strain in strain_values():
                scale = 1.0 + strain
                scale_str = f"{scale:.2f}"

                # Create a copy and modify only one lattice vector
                atoms = atoms0.copy()
                cell = atoms.get_cell().array.copy()
                cell[ax] = cell[ax] * scale  # uniaxial scaling along chosen axis

                # Important: keep fractional coordinates unchanged
                atoms.set_cell(cell, scale_atoms=False)

                out_name = f"orth_{axis_name(ax)}_{scale_str}.vasp"
                out_path = out_subdir / out_name
                write(str(out_path), atoms, format="vasp", vasp5=True, direct=True)

                print(f"        -> axis={axis_name(ax)}, strain={strain:+.2f}, scale={scale_str}, file={out_name}")
                total_written += 1

    print("\n[DONE]")
    print(f"  written: {total_written}")
    print(f"  skipped (missing orth.vasp): {total_skipped_missing}")
    print(f"  skipped (not Direct/Cartesian ERROR): {total_skipped_cart}")
    print(f"  skipped (non-orthogonal): {total_skipped_nonortho}")
    print(f"  output dir: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
