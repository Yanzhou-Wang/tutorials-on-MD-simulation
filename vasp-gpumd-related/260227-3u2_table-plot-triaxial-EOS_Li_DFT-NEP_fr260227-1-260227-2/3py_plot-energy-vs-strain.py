#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

# 设置公共目录
r_dir = "./"  # 结构的目录在当前路径下
w_dir = "./"  # 图片保存路径
fs = 24  # 字体大小
ms = 12  # marker 大小
ss = 24  # 刻度字体大小
lw = 2
mlw = 2.5

# 遍历以 id- 开头的结构目录
for id_dir in os.listdir(r_dir):
    if id_dir.startswith("job_"):
        # 构造DFT和NEP文件路径
        r_f_DFT = os.path.join(r_dir, id_dir, "result-energy-vs-strain_DFT.txt")
        r_f_NEP = os.path.join(r_dir, id_dir, "result-energy-vs-strain_NEP.txt")

        # 检查文件是否存在
        if os.path.exists(r_f_DFT) and os.path.exists(r_f_NEP):
            # ===== 读取三列 =====
            strain_DFT, volume_DFT, energy_DFT = np.loadtxt(r_f_DFT, comments='#', delimiter=None).T
            strain_NEP, volume_NEP, energy_NEP = np.loadtxt(r_f_NEP, comments='#', delimiter=None).T

            # ===== 新增：能量减去最小值 =====
            energy_DFT = energy_DFT - np.min(energy_DFT)
            energy_NEP = energy_NEP - np.min(energy_NEP)

            # 创建 figure 和 ax 对象
            plt.figure(figsize=(8, 6), dpi=200)

            # x 使用 volume
            plt.plot(volume_DFT, energy_DFT, label='DFT', marker='o', linestyle='-', color='#1f77b4',
                     markersize=ms, markerfacecolor='none', linewidth=lw, markeredgewidth=mlw)

            plt.plot(volume_NEP, energy_NEP, label='NEP', marker='+', linestyle='--', color='#ff7f0e',
                     markersize=ms, markerfacecolor='none', linewidth=lw, markeredgewidth=mlw)

            # 设置标题和坐标轴标签
            title_str = id_dir.removeprefix("job_")
            plt.title(title_str, fontsize=fs)
            #plt.title(f'{id_dir}', fontsize=fs)

            plt.xlabel(r'Volume ($\mathrm{\AA}^3$/atom)', fontsize=fs)
            plt.ylabel('Energy (eV/atom)', fontsize=fs)

            # 设置坐标轴刻度的字体大小
            plt.tick_params(axis='both', labelsize=ss)

            # 设置图例
            plt.legend(fontsize=fs)

            # 保存图像为 png 文件
            plt.tight_layout()
            output_file = os.path.join(w_dir, f"fig_{id_dir}.png")
            plt.savefig(output_file)

            # plt.close()

        else:
            print(f"Warning: Missing data files for {id_dir}, skipping this directory.")
