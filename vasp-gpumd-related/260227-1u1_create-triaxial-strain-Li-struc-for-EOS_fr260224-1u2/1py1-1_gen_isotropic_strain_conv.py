#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
from pathlib import Path

# === 用户可调部分 =======================================================
read_comm_dir = Path("../260224-1u2_identify-prim-conv-orth_Li_fr260224-1")

out_dir = Path("./")

in_filename = "conv.vasp"

strain_start = -0.05
strain_end = 0.10
strain_step = 0.01

# 需要处理的结构目录名前缀
job_dir_prefix = "id-mp-"
# =======================================================================

#if out_dir.exists():
#    print(f"[WARN] Removing existing output directory: {out_dir}")
#    shutil.rmtree(out_dir)


def strain_values():
    start_i = int(round(strain_start * 100))
    end_i   = int(round(strain_end * 100))
    step_i  = int(round(strain_step * 100))

    for i in range(start_i, end_i + 1, step_i):
        yield i / 100.0


def main():
    if not read_comm_dir.exists():
        raise FileNotFoundError(f"read_comm_dir not found: {read_comm_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    subdirs = sorted(
        [
            p for p in read_comm_dir.iterdir()
            if p.is_dir() and p.name.startswith(job_dir_prefix)
        ]
    )

    print(f"[INFO] Found {len(subdirs)} structures")

    total_written = 0

    for idx, sd in enumerate(subdirs, 1):
        in_path = sd / in_filename

        if not in_path.exists():
            print(f"[SKIP] ({idx}/{len(subdirs)}) Missing {in_filename} in {sd.name}")
            continue

        lines = in_path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines(True)

        if len(lines) < 2:
            print(f"[SKIP] ({idx}/{len(subdirs)}) File too short: {sd.name}")
            continue

        out_subdir = out_dir / sd.name
        out_subdir.mkdir(parents=True, exist_ok=True)

        print(f"[PROC] ({idx}/{len(subdirs)}) {sd.name}")

        for strain in strain_values():
            scale = 1.0 + strain
            scale_str = f"{scale:.2f}"

            out_path = out_subdir / f"conv_{scale_str}.vasp"

            new_lines = list(lines)
            new_lines[1] = scale_str + "\n"

            out_path.write_text(
                "".join(new_lines),
                encoding="utf-8"
            )

            print(
                f"        -> strain={strain:+.2f}, "
                f"scale={scale_str}, "
                f"file={out_path.name}"
            )

            total_written += 1

    print(f"\n[DONE] Written {total_written} structures into: {out_dir.resolve()}")


if __name__ == "__main__":  
    main()
