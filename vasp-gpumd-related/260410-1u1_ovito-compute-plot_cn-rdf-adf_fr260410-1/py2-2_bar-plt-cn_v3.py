#!/usr/bin/env python3

'''
Plot CN distribution bar chart from result-cn.txt files.

Usage:
    Put this script in the parent directory containing job_* folders, then run:
        ./py_plot-cn.py
'''

# ============================================================
# User-defined parameters
# ============================================================
r_jb_p_dir = "./"
job_start_str = "job_"
r_f_n = "result-cn.txt"
w_f_n = "fig_cn.png"

fs = 22
figsize = (8, 6)
dpi = 200
# ============================================================


import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def parse_type_mapping(txt_file):
    """
    Parse type mapping from comment line, e.g.
    # type mapping: 1=C, 2=H

    Returns:
        dict like {1: 'C', 2: 'H'}
    """
    with open(txt_file, "r") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("#"):
                continue

            if s.startswith("# type mapping:"):
                mapping_str = s.split(":", 1)[1].strip()
                mapping = {}

                if mapping_str:
                    items = [x.strip() for x in mapping_str.split(",")]
                    for item in items:
                        if "=" in item:
                            tid, elem = item.split("=")
                            mapping[int(tid.strip())] = elem.strip()

                return mapping

    raise ValueError(f"Cannot find type mapping in file: {txt_file}")


def build_math_label(elem, cn):
    """
    Build x tick label in math style, e.g. C_{CN=3}
    """
    return rf"${elem}_{{\mathrm{{CN}}={cn}}}$"


def Proc_plot_cn(cn_file, fig_file, title_str):
    """
    Plot CN distribution bar chart.
    """
    type_mapping = parse_type_mapping(cn_file)

    data = np.loadtxt(cn_file, comments="#")
    data = np.atleast_2d(data)

    if data.shape[1] < 6:
        raise ValueError(f"CN data file must have at least 6 columns: {cn_file}")

    particle_types = data[:, 1].astype(int)
    coord_numbers = data[:, 5].astype(int)

    # Total number of particles in the system
    N = data.shape[0]

    # Count occurrences of each (element, CN)
    counter = Counter()
    for ptype, cn in zip(particle_types, coord_numbers):
        elem = type_mapping.get(ptype, str(ptype))
        counter[(elem, cn)] += 1

    # Sort by element first, then CN
    sorted_items = sorted(counter.items(), key=lambda x: (x[0][0], x[0][1]))

    labels = [build_math_label(elem, cn) for (elem, cn), count in sorted_items]
    counts = [count for (elem, cn), count in sorted_items]

    x = np.arange(len(labels))

    # Different colors and hatches for better visual distinction
    color_cycle = [
        "tab:blue", "tab:orange", "tab:green", "tab:red",
        "tab:purple", "tab:brown", "tab:pink", "tab:gray",
        "tab:olive", "tab:cyan"
    ]
    hatch_cycle = ["", "//", "\\\\", "xx", "--", "||", "++", "..", "oo", "**"]

    plt.figure(figsize=figsize)

    bars = []
    for i, (xi, count) in enumerate(zip(x, counts)):
        bar = plt.bar(
            xi,
            count,
            color=color_cycle[i % len(color_cycle)],
            hatch=hatch_cycle[i % len(hatch_cycle)],
            edgecolor="black",
            linewidth=1.0
        )
        bars.append(bar[0])

    plt.xticks(x, labels, rotation=45, ha="right", fontsize=fs)
    plt.yticks(fontsize=fs)

    plt.ylabel("Count", fontsize=fs)
    plt.title(title_str, fontsize=fs)

    # Add count/N labels on top of each bar
    ymax = max(counts) if counts else 1
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01 * ymax,
            f"{count}/{N}",
            ha="center",
            va="bottom",
            fontsize=fs * 0.65,
            rotation=0
        )

    plt.tight_layout()
    plt.savefig(fig_file, dpi=dpi)
    #plt.close()


def main():
    dirs = [
        dir_name for dir_name in os.listdir(r_jb_p_dir)
        if dir_name.startswith(job_start_str)
        and os.path.isdir(os.path.join(r_jb_p_dir, dir_name))
    ]

    for dir_name in dirs:
        r_dest_file = os.path.join(r_jb_p_dir, dir_name, r_f_n)
        w_dest_file = os.path.join(r_jb_p_dir, dir_name, w_f_n)

        if not os.path.exists(r_dest_file):
            print(f"[WARN] File not found: {r_dest_file}")
            continue

        try:
            Proc_plot_cn(
                cn_file=r_dest_file,
                fig_file=w_dest_file,
                title_str=dir_name
            )
            print(f"{dir_name} is done ...")
        except Exception as e:
            print(f"[ERROR] Failed for {dir_name}")
            print(f"        file: {r_dest_file}")
            print(f"        reason: {e}")


if __name__ == '__main__':
    main()