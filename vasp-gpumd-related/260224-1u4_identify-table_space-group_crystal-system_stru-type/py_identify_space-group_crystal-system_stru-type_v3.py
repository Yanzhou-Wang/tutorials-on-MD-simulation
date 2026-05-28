#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Identify crystal system, space group and structure type for VASP cell structures.

This script uses:
  1) pymatgen.symmetry.analyzer.SpacegroupAnalyzer
     for space group and crystal system

  2) pymatgen.analysis.prototypes.AflowPrototypeMatcher
     for structure type / prototype matching

Output:
  A table file, where each row corresponds to one structure.
"""

import os

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

try:
    from pymatgen.analysis.prototypes import AflowPrototypeMatcher
    HAS_AFLOW_MATCHER = True
except Exception:
    HAS_AFLOW_MATCHER = False


# === 用户可调部分 =======================================================
# 主目录（相对当前脚本运行位置）
main_path = "../260224-1u2_identify-prim-conv-orth_Li_fr260224-1"

# 用户需要处理的算例目录（这个要用于与 main_path 的拼接）
job_items = [
    "id-mp-604313_Li",
]

# 用户指定输入结构文件名（一定要是晶胞！！！）
in_file = "conv.vasp"

# 用户指定输出文件名
out_file = "result-structure-symmetry.txt"

# 对称性判断参数
symprec = 1e-3
angle_tolerance = 5.0
# =======================================================================


def get_spacegroup_and_crystal_system(structure, symprec=1e-3, angle_tolerance=5.0):
    """
    Use pymatgen SpacegroupAnalyzer to get:
      - space group symbol
      - space group number
      - crystal system
    """
    sga = SpacegroupAnalyzer(
        structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance
    )

    sg_symbol = sga.get_space_group_symbol()
    sg_number = sga.get_space_group_number()
    crystal_system = sga.get_crystal_system()

    space_group = f"{sg_symbol} ({sg_number})"

    return space_group, crystal_system


def extract_prototype_string(proto):
    """
    Convert one prototype dict returned by AflowPrototypeMatcher into a readable string.

    Different pymatgen versions may return slightly different dict keys.
    This function tries several commonly used keys.
    """
    if proto is None:
        return "unknown"

    if isinstance(proto, str):
        return proto

    if not isinstance(proto, dict):
        return str(proto)

    # Common useful fields in pymatgen prototype dictionaries.
    preferred_keys = [
        "prototype",
        "aflow",
        "aflow_prototype",
        "strukturbericht",
        "mineral",
        "tags",
    ]

    for key in preferred_keys:
        if key in proto and proto[key] not in (None, "", []):
            value = proto[key]

            if isinstance(value, list):
                return ",".join(str(x) for x in value)

            return str(value)

    # If no preferred field is found, build a compact fallback string.
    compact_items = []
    for key, value in proto.items():
        if value in (None, "", []):
            continue
        compact_items.append(f"{key}={value}")

    if compact_items:
        return "; ".join(compact_items)

    return "unknown"


def get_structure_type_by_aflow_matcher(structure):
    """
    Use pymatgen AflowPrototypeMatcher to identify structure type.

    If no prototype is found, return 'unknown'.
    """
    if not HAS_AFLOW_MATCHER:
        return "unknown"

    try:
        matcher = AflowPrototypeMatcher()

        prototypes = matcher.get_prototypes(structure)

        if not prototypes:
            return "unknown"

        # Usually the first matched prototype is the best candidate.
        proto0 = prototypes[0]
        return extract_prototype_string(proto0)

    except Exception as e:
        return f"unknown"


def analyze_one_structure(job_name, structure_path):
    """
    Analyze one VASP structure file.
    """
    structure = Structure.from_file(structure_path)

    space_group, crystal_system = get_spacegroup_and_crystal_system(
        structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance
    )

    structure_type = get_structure_type_by_aflow_matcher(structure)

    return {
        "name": job_name,
        "space_group": space_group,
        "crystal_system": crystal_system,
        "structure_type": structure_type,
    }


def write_table(results, out_path):
    """
    Write results into a formatted table.
    """
    headers = [
        "Structure-name",
        "Space-group",
        "Crystal-system",
        "Structure-type",
    ]

    rows = []
    for r in results:
        rows.append([
            r["name"],
            r["space_group"],
            r["crystal_system"],
            r["structure_type"],
        ])

    all_rows = [headers] + rows

    widths = []
    for icol in range(len(headers)):
        width = max(len(str(row[icol])) for row in all_rows)
        widths.append(width)

    with open(out_path, "w") as f:
        header_line = "  ".join(
            str(headers[i]).ljust(widths[i]) for i in range(len(headers))
        )
        f.write(header_line + "\n")

        sep_line = "  ".join("-" * widths[i] for i in range(len(headers)))
        f.write(sep_line + "\n")

        for row in rows:
            line = "  ".join(
                str(row[i]).ljust(widths[i]) for i in range(len(headers))
            )
            f.write(line + "\n")


def main():
    if not HAS_AFLOW_MATCHER:
        print("[WARN] Cannot import AflowPrototypeMatcher from pymatgen.")
        print("[WARN] Structure-type will be written as unknown.")

    results = []

    for job_name in job_items:
        structure_path = os.path.join(main_path, job_name, in_file)

        if not os.path.isfile(structure_path):
            print(f"[WARN] File not found: {structure_path}")
            results.append({
                "name": job_name,
                "space_group": "unknown",
                "crystal_system": "unknown",
                "structure_type": "unknown",
            })
            continue

        try:
            print(f"[INFO] Processing: {structure_path}")
            result = analyze_one_structure(job_name, structure_path)
            results.append(result)

        except Exception as e:
            print(f"[ERROR] Failed to process: {structure_path}")
            print(f"        Reason: {e}")
            results.append({
                "name": job_name,
                "space_group": "unknown",
                "crystal_system": "unknown",
                "structure_type": "unknown",
            })

    write_table(results, out_file)

    print(f"\nDone. Results written to: {out_file}")


if __name__ == "__main__":
    main()