#!/usr/bin/env python3

'''
Plot partial RDF curves from result-rdf.txt files.

Usage:
    Put this script in the parent directory containing job_* folders, then run:
        ./py_plot-rdf.py
'''

# ============================================================
# User-defined parameters
# ============================================================
r_jb_p_dir = "./"
job_start_str = "job_"
r_f_n = "result-rdf.txt"
w_f_n = "fig_rdf.png"

fs = 22
lw = 2.2
figsize = (8, 6)
dpi = 200
# ============================================================


import os
import numpy as np
import matplotlib.pyplot as plt


def parse_rdf_pair_labels(txt_file):
    """
    Parse RDF pair labels from comment lines in result-rdf.txt.
    """
    with open(txt_file, "r") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("#"):
                continue

            if "Pair separation distance" in s:
                line_clean = s.lstrip("#").strip()
                first_quote = line_clean.find('"')
                second_quote = line_clean.find('"', first_quote + 1)

                if first_quote != -1 and second_quote != -1:
                    rest = line_clean[second_quote + 1:].strip()
                    if rest:
                        return rest.split()

    raise ValueError(f"Cannot find RDF pair-label header in file: {txt_file}")


def Proc_plot_rdf(rdf_file, fig_file, title_str):
    """
    Load RDF data and plot all pair curves in one figure.
    """
    pair_labels = parse_rdf_pair_labels(rdf_file)

    data = np.loadtxt(rdf_file, comments="#")
    data = np.atleast_2d(data)

    if data.shape[1] < 2:
        raise ValueError(f"RDF data file must have at least 2 columns: {rdf_file}")

    r = data[:, 0]
    y = data[:, 1:]

    if y.shape[1] != len(pair_labels):
        raise ValueError(
            f"Number of parsed pair labels ({len(pair_labels)}) does not match "
            f"number of RDF columns ({y.shape[1]}) in file: {rdf_file}"
        )

    plt.figure(figsize=figsize)

    for i, pair in enumerate(pair_labels):
        plt.plot(r, y[:, i], linewidth=lw, label=pair)

    plt.xlabel(r"$r$ ($\mathrm{\AA}$)", fontsize=fs)
    plt.ylabel(r"$g(r)$", fontsize=fs)

    # ✅ 新增 title
    plt.title(title_str, fontsize=fs)

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    plt.legend(fontsize=fs)
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
            # ✅ 这里把目录名作为 title 传入
            Proc_plot_rdf(
                rdf_file=r_dest_file,
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
    