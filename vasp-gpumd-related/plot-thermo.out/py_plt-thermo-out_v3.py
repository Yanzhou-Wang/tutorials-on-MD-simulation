#!/usr/bin/env python3

'''
Usage: 
    go the gpumd case directory where you can see ouput `thermo.out`, inputs `run.in` and `model.xyz`, and run the script:
    `./py_plot-thermo-out_v3.py`           
    PS: the script gets number of system particles from `model.xyz`, 
        get running time info from `run.in`,
        and plot thermo-realted quantities by reading `thermo.out`
'''

import os
import numpy as np
import matplotlib.pyplot as plt

# ===== plotting parameters =====
fs = 22
lw = 2.2

# ===== filenames =====
runin_file = 'run.in'
thermo_file = 'thermo.out'
model_file = 'model.xyz'


# ============================================================
# 1. 解析 run.in
# ============================================================
def parse_run_in(filename):
    modules = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    current = {}
    last_time_step = None

    for line in lines:
        # 去掉注释
        line = line.split('#')[0].strip()
        if not line:
            continue

        tokens = line.split()
        key = tokens[0]

        # 模块开始
        if key == 'ensemble':
            if current:
                modules.append(current)
            current = {}
            current['ensemble'] = tokens[1:]

        elif key == 'time_step':
            current['time_step'] = float(tokens[1])
            last_time_step = float(tokens[1])

        elif key == 'dump_thermo':
            current['dump_thermo'] = int(tokens[1])

        elif key == 'run':
            current['run'] = int(tokens[1])

            # time_step 继承
            if 'time_step' not in current:
                if last_time_step is None:
                    raise ValueError("time_step 未定义且无法继承")
                current['time_step'] = last_time_step

            modules.append(current)
            current = {}

    return modules


# ============================================================
# 2. 根据 modules 构造时间序列
# ============================================================
def build_time_array(modules):
    time_list = []
    t_accum = 0.0  # fs

    for m in modules:
        dt = m['time_step']           # fs
        dump = m['dump_thermo']
        run = m['run']

        # 输出点数量（整数部分）
        n_points = run // dump

        # 每个点之间的时间间隔
        dt_dump = dump * dt  # fs

        for k in range(1, n_points + 1):
            t = t_accum + k * dt_dump
            time_list.append(t)

        # 更新时间累计（整个模块的真实时间）
        t_accum += run * dt

    # 转换为 ps
    time_array = np.array(time_list) / 1000.0
    return time_array


# ============================================================
# 3. 读取 model.xyz 第一行的原子数
# ============================================================
def read_natoms_from_model_xyz(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"{filename} 不存在")

    with open(filename, 'r') as f:
        first_line = f.readline().strip()

    try:
        natoms = int(first_line)
    except ValueError:
        raise ValueError(f"{filename} 第一行不是合法的原子数: {first_line}")

    return natoms


# ============================================================
# 4. 主程序
# ============================================================
def main():

    # ---------- 读取 run.in ----------
    if not os.path.isfile(runin_file):
        raise FileNotFoundError("run.in 不存在")

    modules = parse_run_in(runin_file)

    # ---------- 构造时间 ----------
    time = build_time_array(modules)

    # ---------- 读取 thermo.out ----------
    if not os.path.isfile(thermo_file):
        raise FileNotFoundError("thermo.out 不存在")

    thermo = np.loadtxt(thermo_file)
    T = thermo[:, 0]
    U_total = thermo[:, 2]

    # ---------- 压强分量 ----------
    Pxx = thermo[:, 3]
    Pyy = thermo[:, 4]
    Pzz = thermo[:, 5]
    Pyz = thermo[:, 6]
    Pxz = thermo[:, 7]
    Pxy = thermo[:, 8]

    # ---------- 晶格基矢分量 ----------
    ax = thermo[:, 9]
    ay = thermo[:, 10]
    az = thermo[:, 11]
    bx = thermo[:, 12]
    by = thermo[:, 13]
    bz = thermo[:, 14]
    cx = thermo[:, 15]
    cy = thermo[:, 16]
    cz = thermo[:, 17]

    # ---------- 计算晶格常数 a, b, c ----------
    a = np.sqrt(ax**2 + ay**2 + az**2)
    b = np.sqrt(bx**2 + by**2 + bz**2)
    c = np.sqrt(cx**2 + cy**2 + cz**2)

    # ---------- 读取原子数 ----------
    natoms = read_natoms_from_model_xyz(model_file)
    U = U_total / natoms

    # ---------- 长度检查 ----------
    if len(time) != len(T):
        print("WARNING:")
        print(f"  时间点数 = {len(time)}")
        print(f"  thermo 行数 = {len(T)}")

        # 取最小长度（保证绘图不中断）
        n = min(len(time), len(T))
        time = time[:n]
        T = T[:n]
        U = U[:n]

        Pxx = Pxx[:n]
        Pyy = Pyy[:n]
        Pzz = Pzz[:n]
        Pyz = Pyz[:n]
        Pxz = Pxz[:n]
        Pxy = Pxy[:n]

        a = a[:n]
        b = b[:n]
        c = c[:n]

    # ---------- 绘制温度 ----------
    plt.figure(figsize=(8, 6))
    plt.plot(time, T, '-', linewidth=lw, color='tab:blue')

    plt.xlabel(r'$t$ (ps)', fontsize=fs)
    plt.ylabel(r'$T$ (K)', fontsize=fs)

    plt.xticks(fontsize=fs-3)
    plt.yticks(fontsize=fs-3)

    # plt.grid(True)
    plt.tight_layout()

    plt.savefig('fig_temperature_vs_time.png', dpi=200)
    print('fig_temperature_vs_time.png saved ...')
    # plt.show()
    # plt.close()

    # ---------- 绘制势能 ----------
    plt.figure(figsize=(8, 6))
    plt.plot(time, U, '-', linewidth=lw, color='tab:red')

    plt.xlabel(r'$t$ (ps)', fontsize=fs)
    plt.ylabel(r'$U$ (eV/atom)', fontsize=fs)

    plt.xticks(fontsize=fs-3)
    plt.yticks(fontsize=fs-3)

    # plt.grid(True)
    plt.tight_layout()

    plt.savefig('fig_potential_energy_vs_time.png', dpi=200)
    print('fig_potential_energy_vs_time.png saved ...')
    # plt.show()
    # plt.close()

    # ---------- 绘制压强分量 ----------
    plt.figure(figsize=(8, 6))
    plt.plot(time, Pxx, '-', linewidth=lw, label='Pxx')
    plt.plot(time, Pyy, '-.', linewidth=lw, label='Pyy')
    plt.plot(time, Pzz, '--', linewidth=lw, label='Pzz')
    plt.plot(time, Pyz, '-', linewidth=lw, label='Pyz')
    plt.plot(time, Pxz, '-.', linewidth=lw, label='Pxz')
    plt.plot(time, Pxy, '--', linewidth=lw, label='Pxy')

    plt.xlabel(r'$t$ (ps)', fontsize=fs)
    plt.ylabel(r'$P$ (GPa)', fontsize=fs)

    plt.xticks(fontsize=fs-3)
    plt.yticks(fontsize=fs-3)
    plt.legend(fontsize=fs-4)

    # plt.grid(True)
    plt.tight_layout()

    plt.savefig('fig_pressure_vs_time.png', dpi=200)
    print('fig_pressure_vs_time.png saved ...')
    # plt.show()
    # plt.close()

    # ---------- 绘制晶格常数 ----------
    plt.figure(figsize=(8, 6))
    plt.plot(time, a, '-', linewidth=lw, label='a')
    plt.plot(time, b, '-.', linewidth=lw, label='b')
    plt.plot(time, c, '--', linewidth=lw, label='c')

    plt.xlabel(r'$t$ (ps)', fontsize=fs)
    plt.ylabel(r'Lattice ($\mathrm{\AA}$)', fontsize=fs)

    plt.xticks(fontsize=fs-3)
    plt.yticks(fontsize=fs-3)
    plt.legend(fontsize=fs-4)

    # plt.grid(True)
    plt.tight_layout()

    plt.savefig('fig_lattice_vs_time.png', dpi=200)
    print('fig_lattice_vs_time.png saved ...')
    # plt.show()
    # plt.close()


# ============================================================
if __name__ == "__main__":
    main()