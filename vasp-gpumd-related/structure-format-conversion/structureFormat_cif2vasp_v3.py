#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# py_cif2vasp.py
#
# Usage:
#   1) Recursively convert all *.cif under current directory (in-place) and write it in same place where x.cif is:
#        ./py_cif2vasp.py
#        ./py_cif2vasp.py ./
#
#   2) Convert under a given path (in-place) and write it in same place:
#        ./py_cif2vasp.py /path/to/root_dir
#        ./py_cif2vasp.py /path/to/file.cif
#
#   3) Convert from READ_PATH and write outputs into WRITE_PATH:
#        ./py_cif2vasp.py read_path write_path
#
#      - If read_path is a .cif file:
#          write_path/<basename>.vasp
#      - If read_path is a directory:
#          recursively find **/*.cif under read_path, and write to write_path
#          keeping relative subdirectories (to avoid name conflicts).
#
#    4) ../py_cif2vasp.py ../Ta2Ni_mp-1101992_primitive.cif ./
#
# Output style (same as your original):
#   write(..., format="vasp", vasp5=True, direct=True, sort=False)
#

import sys
from pathlib import Path
from ase.io import read, write


def cif_to_vasp(cif_path: Path, out_path: Path):
    atoms = read(str(cif_path))
    write(str(out_path), atoms, format="vasp", vasp5=True, direct=True, sort=False)


def main():
    args = sys.argv[1:]

    # Case A: no args -> in-place recursive under current dir
    if len(args) == 0:
        read_root = Path(".").resolve()
        write_root = None  # in-place

    # Case B: one arg -> if dir: in-place recursive; if file: in-place single
    elif len(args) == 1:
        p = Path(args[0]).expanduser().resolve()
        if not p.exists():
            print(f"ERROR: path does not exist: {p}")
            sys.exit(1)
        read_root = p
        write_root = None  # in-place

    # Case C: two args -> read_path + write_path
    elif len(args) == 2:
        read_root = Path(args[0]).expanduser().resolve()
        write_root = Path(args[1]).expanduser().resolve()

        if not read_root.exists():
            print(f"ERROR: read_path does not exist: {read_root}")
            sys.exit(1)
        if not write_root.exists() or not write_root.is_dir():
            print(f"ERROR: write_path must be an existing directory: {write_root}")
            sys.exit(1)

    else:
        print("ERROR: too many arguments.")
        print("See usage in script header comments.")
        sys.exit(1)

    ok, fail = 0, 0

    # If read_root is a single CIF file
    if read_root.is_file():
        if read_root.suffix.lower() != ".cif":
            print(f"ERROR: input file is not .cif: {read_root}")
            sys.exit(1)

        if write_root is None:
            out_path = read_root.with_suffix(".vasp")  # in-place
        else:
            out_path = write_root / (read_root.stem + ".vasp")

        try:
            cif_to_vasp(read_root, out_path)
            print(f"OK   {read_root} -> {out_path}")
            ok += 1
        except Exception as e:
            print(f"FAIL {read_root}\n     Reason: {e}")
            fail += 1

    # If read_root is a directory: recursive
    else:
        cif_files = sorted(read_root.rglob("*.cif"))
        if not cif_files:
            print(f"No *.cif found under: {read_root}")
            sys.exit(1)

        print(f"Read:  {read_root}")
        print(f"Mode:  {'in-place' if write_root is None else 'write to: ' + str(write_root)}")
        print(f"Found {len(cif_files)} CIF file(s). Converting...\n")

        for i, cif_path in enumerate(cif_files, start=1):
            try:
                if write_root is None:
                    out_path = cif_path.with_suffix(".vasp")  # in-place
                else:
                    rel = cif_path.relative_to(read_root)     # keep relative path
                    out_path = (write_root / rel).with_suffix(".vasp")
                    out_path.parent.mkdir(parents=True, exist_ok=True)

                cif_to_vasp(cif_path, out_path)
                print(f"[{i:04d}/{len(cif_files):04d}] OK   {cif_path} -> {out_path}")
                ok += 1
            except Exception as e:
                print(f"[{i:04d}/{len(cif_files):04d}] FAIL {cif_path}\n"
                      f"                      Reason: {e}")
                fail += 1

    print(f"\nDone. Success: {ok}, Failed: {fail}")
    sys.exit(0 if fail == 0 else 2)


if __name__ == "__main__":
    main()
