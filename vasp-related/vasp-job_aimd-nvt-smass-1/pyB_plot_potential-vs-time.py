#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot potential energy (E_pot, TOTEN) vs time(fs) from result-potential-vs-time.txt
using np.loadtxt.

Usage:
    python plot_potential-vs-time.py result-potential-vs-time.txt
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) > 1:
        infile = sys.argv[1]
    else:
        infile = "result-potential-vs-time.txt"

    # 读取两列数据（time, potential）
    data = np.loadtxt(infile, comments="#")
    time = data[:, 0]
    pot  = data[:, 1]

    # 绘图参数
    fs = 18        # fontsize
    lw = 1.8       # linewidth
    ms = 8         # markersize

    plt.figure(figsize=(8, 6), dpi=200)

    # 空心 marker，黑色线条
    plt.plot(time, pot,
             '-o',
             lw=lw,
             markersize=ms,
             color='blue',
             markerfacecolor='none')

    plt.xlabel("Time (fs)", fontsize=fs)
    plt.ylabel("Potential energy (eV)", fontsize=fs)
    plt.title("AIMD Potential Energy vs Time", fontsize=fs)

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    plt.tight_layout()

    outname = "fig_potential-vs-time.png"
    plt.savefig(outname, dpi=200)
#    plt.close()

    print(f"Done. Plot saved as '{outname}'.")


if __name__ == "__main__":
    main()
