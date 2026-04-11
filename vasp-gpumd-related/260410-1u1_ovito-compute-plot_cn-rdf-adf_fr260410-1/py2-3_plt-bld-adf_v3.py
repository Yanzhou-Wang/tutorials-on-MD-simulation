#!/usr/bin/env python3

'''
Plot bond-length distribution (BLD) and bond-angle distribution (ADF)

Usage:
    Put this script in the parent directory containing job_* folders, then run:
        ./py_plot-bld-adf.py
'''

# ============================================================
# User-defined parameters
# ============================================================
r_jb_p_dir = "./"
job_start_str = "job_"

r_bld_f_n = "result-bld.txt"
r_adf_f_n = "result-adf.txt"

w_bld_f_n = "fig_bld.png"
w_adf_f_n = "fig_adf.png"

fs = 22
lw = 2.2
figsize = (8, 6)
dpi = 200
# ============================================================


import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Robust header parser (FIXED VERSION)
# ============================================================
def parse_pair_labels(txt_file):
    """
    Parse pair labels from header line.

    Supports:
        RDF:  # "Pair separation distance" C-C C-H H-H
        BLD:  # Length C-C C-H H-H
        ADF:  # Angle C-C-C C-C-H ...
    """
    with open(txt_file, "r") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("#"):
                continue

            line_clean = s.lstrip("#").strip()

            # ---------- Case 1: RDF style ----------
            if '"' in line_clean:
                first_quote = line_clean.find('"')
                second_quote = line_clean.find('"', first_quote + 1)

                if first_quote != -1 and second_quote != -1:
                    rest = line_clean[second_quote + 1:].strip()
                    if rest:
                        return rest.split()

            # ---------- Case 2: BLD / ADF ----------
            tokens = line_clean.split()

            if len(tokens) >= 2:
                if tokens[0] in ["Length", "Angle"]:
                    return tokens[1:]

    raise ValueError(f"Cannot find pair labels in file: {txt_file}")


# ============================================================
# Generic plotting function
# ============================================================
def plot_distribution(txt_file, fig_file, xlabel, ylabel, title_str):

    pair_labels = parse_pair_labels(txt_file)

    data = np.loadtxt(txt_file, comments="#")
    data = np.atleast_2d(data)

    x = data[:, 0]
    y = data[:, 1:]

    if y.shape[1] != len(pair_labels):
        raise ValueError(f"Mismatch between data columns and labels in {txt_file}")

    plt.figure(figsize=figsize)

    for i, pair in enumerate(pair_labels):
        plt.plot(x, y[:, i], linewidth=lw, label=pair)

    plt.xlabel(xlabel, fontsize=fs)
    plt.ylabel(ylabel, fontsize=fs)

    plt.title(title_str, fontsize=fs)

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig(fig_file, dpi=dpi)
    #plt.close()


# ============================================================
# Main
# ============================================================
def main():

    dirs = [
        d for d in os.listdir(r_jb_p_dir)
        if d.startswith(job_start_str)
        and os.path.isdir(os.path.join(r_jb_p_dir, d))
    ]

    for dir_name in dirs:

        # -------- BLD --------
        bld_file = os.path.join(r_jb_p_dir, dir_name, r_bld_f_n)
        fig_bld = os.path.join(r_jb_p_dir, dir_name, w_bld_f_n)

        if os.path.exists(bld_file):
            try:
                plot_distribution(
                    txt_file=bld_file,
                    fig_file=fig_bld,
                    xlabel = r"$\ell$ ($\mathrm{\AA}$)",
                    ylabel = r"$P(\ell)$",
                    title_str=dir_name
                )
            except Exception as e:
                print(f"[ERROR] BLD failed: {dir_name}")
                print(e)
        else:
            print(f"[WARN] Missing {bld_file}")

        # -------- ADF --------
        adf_file = os.path.join(r_jb_p_dir, dir_name, r_adf_f_n)
        fig_adf = os.path.join(r_jb_p_dir, dir_name, w_adf_f_n)

        if os.path.exists(adf_file):
            try:
                plot_distribution(
                    txt_file=adf_file,
                    fig_file=fig_adf,
                    xlabel=r"$\theta$ ($^\circ$)",
                    ylabel=r"$\phi(\theta)$",
                    title_str=dir_name
                )
            except Exception as e:
                print(f"[ERROR] ADF failed: {dir_name}")
                print(e)
        else:
            print(f"[WARN] Missing {adf_file}")

        print(f"{dir_name} is done ...")


if __name__ == '__main__':
    main()
    
    