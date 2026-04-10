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
lw = 2   # 线宽
mlw = 2.5  # marker 轮廓线宽度

# 定义颜色列表
DFT_colors = ['b', 'g', 'c', 'm']  # DFT 用的颜色
NEP_colors = ['r', 'y', 'orange', 'purple']  # NEP 用的颜色

# 遍历以 id- 开头的结构目录
for id_dir in os.listdir(r_dir):
    if id_dir.startswith("id-"):
        # 检查结构目录下的文件，判断有哪些轴的计算结果
        files = os.listdir(os.path.join(r_dir, id_dir))
        
        # 判断哪些轴有数据
        axes = []
        if "result-energy-vs-strain_a_DFT.txt" in files and "result-energy-vs-strain_a_NEP.txt" in files:
            axes.append('a')
        if "result-energy-vs-strain_b_DFT.txt" in files and "result-energy-vs-strain_b_NEP.txt" in files:
            axes.append('b')
        if "result-energy-vs-strain_c_DFT.txt" in files and "result-energy-vs-strain_c_NEP.txt" in files:
            axes.append('c')

        # ========= 新增：先找该结构下 DFT 和 NEP 的全局最小能量 =========
        e_min_DFT = None
        e_min_NEP = None

        for axis in axes:
            r_f_DFT = os.path.join(r_dir, id_dir, f"result-energy-vs-strain_{axis}_DFT.txt")
            r_f_NEP = os.path.join(r_dir, id_dir, f"result-energy-vs-strain_{axis}_NEP.txt")

            if os.path.exists(r_f_DFT):
                strain_tmp, energy_tmp = np.loadtxt(r_f_DFT, comments='#', delimiter=None).T
                this_min = np.min(energy_tmp)
                if e_min_DFT is None or this_min < e_min_DFT:
                    e_min_DFT = this_min

            if os.path.exists(r_f_NEP):
                strain_tmp, energy_tmp = np.loadtxt(r_f_NEP, comments='#', delimiter=None).T
                this_min = np.min(energy_tmp)
                if e_min_NEP is None or this_min < e_min_NEP:
                    e_min_NEP = this_min
        # ==========================================================

        # 创建 figure 和 ax 对象
        plt.figure(figsize=(8, 6), dpi=200)

        # 设置不同的 marker 对应不同的轴
        marker_DFT = {'a': 'o', 'b': 's', 'c': 'D'}  # DFT 轴的 marker：圆形，方形，菱形
        marker_NEP = {'a': '+', 'b': 'x', 'c': '*'}  # NEP 轴的 marker：加号，叉号，星号

        # 根据不同轴的情况绘制不同的数据
        dft_color_index = 0
        nep_color_index = 0
        
        for axis in axes:
            r_f_DFT = os.path.join(r_dir, id_dir, f"result-energy-vs-strain_{axis}_DFT.txt")
            r_f_NEP = os.path.join(r_dir, id_dir, f"result-energy-vs-strain_{axis}_NEP.txt")

            if os.path.exists(r_f_DFT) and os.path.exists(r_f_NEP):
                # 使用 numpy 加载数据
                strain_DFT, energy_DFT = np.loadtxt(r_f_DFT, comments='#', delimiter=None).T
                strain_NEP, energy_NEP = np.loadtxt(r_f_NEP, comments='#', delimiter=None).T

                # ========= 新增：所有能量减去各自体系的全局最小值 =========
                energy_DFT = energy_DFT - e_min_DFT
                energy_NEP = energy_NEP - e_min_NEP
                # =====================================================

                # 根据轴确定线型和颜色
                label_DFT = f"DFT-{axis}"
                label_NEP = f"NEP-{axis}"
                linestyle_DFT = '-'  # DFT 用实线
                linestyle_NEP = '--'  # NEP 用虚线

                # 绘制 DFT 数据
                plt.plot(strain_DFT, energy_DFT, label=label_DFT, marker=marker_DFT[axis], linestyle=linestyle_DFT, color=DFT_colors[dft_color_index], markersize=ms, markerfacecolor='none', linewidth=lw, markeredgewidth=mlw)
                
                # 绘制 NEP 数据
                plt.plot(strain_NEP, energy_NEP, label=label_NEP, marker=marker_NEP[axis], linestyle=linestyle_NEP, color=NEP_colors[nep_color_index], markersize=ms, markerfacecolor='none', linewidth=lw, markeredgewidth=mlw)
                
                # 更新颜色索引，避免超出范围
                dft_color_index += 1
                nep_color_index += 1
                if dft_color_index >= len(DFT_colors):  # 如果颜色用完了，重置
                    dft_color_index = 0
                if nep_color_index >= len(NEP_colors):  # 如果颜色用完了，重置
                    nep_color_index = 0

        # 设置标题和坐标轴标签
        plt.title(f'{id_dir}', fontsize=fs)
        plt.xlabel('Strain', fontsize=fs)
        plt.ylabel('Energy (eV/atom)', fontsize=fs)

        # 设置坐标轴刻度的字体大小
        plt.tick_params(axis='both', labelsize=ss)

        # 设置图例
        plt.legend(fontsize=fs)

        # 保存图像为 png 文件
        plt.tight_layout()
        output_file = os.path.join(w_dir, f"fig_{id_dir}.png")
        plt.savefig(output_file)
        
        # 关闭当前图形
        #plt.close()

    else:
        print(f"Warning: Missing data files for {id_dir}, skipping this directory.")