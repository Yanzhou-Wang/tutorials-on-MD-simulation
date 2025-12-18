#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot T(K) vs time(fs) from result-T-vs-time.txt using np.loadtxt.

Usage:
    python plot_T_vs_time.py result-T-vs-time.txt
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) > 1:
        infile = sys.argv[1]
    else:
        infile = "result-T-vs-time.txt"

    # 用 np.loadtxt 读取，忽略注释行
    data = np.loadtxt(infile, comments="#")
    time = data[:, 0]
    temp = data[:, 1]

    # 绘图参数
    fs = 18        # fontsize
    lw = 1.8       # linewidth
    ms = 8         # markersize

    plt.figure(figsize=(8, 6), dpi=200)

    # 空心 marker
    plt.plot(time, temp,
             '-o',
             lw=lw,
             markersize=ms,
             color='black',
             markerfacecolor='none')

    plt.xlabel("Time (fs)", fontsize=fs)
    plt.ylabel("Temperature (K)", fontsize=fs)
    plt.title("AIMD Temperature vs Time", fontsize=fs)

    # 坐标轴刻度
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    plt.tight_layout()

    outname = "fig_T-vs-time.png"
    plt.savefig(outname, dpi=200)
#    plt.close()

    print(f"Done. Plot saved as '{outname}'.")


if __name__ == "__main__":
    main()
