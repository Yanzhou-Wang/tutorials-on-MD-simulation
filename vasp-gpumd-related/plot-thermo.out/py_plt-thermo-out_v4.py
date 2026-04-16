#!/usr/bin/env python3

'''
Usage: 
    go the gpumd case directory where you can see outputs `thermo.out`, inputs `run.in` and [`model.xyz`], and run the script:
    `./py_plot-thermo-out_v3.py`           
    PS: the script [gets number of system particles from `model.xyz` for potential energy per particle] (如果`model.xyz`缺失，则potential energy是总能), 
        gets running time info from `run.in`,
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
# 1. parse run.in
# ============================================================
def parse_run_in(filename):

    modules = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    current = {}
    last_time_step = None

    for line in lines:
        line = line.split('#')[0].strip()
        if not line:
            continue

        tokens = line.split()
        key = tokens[0]

        if key == 'ensemble':
            current['ensemble'] = tokens[1:]

        elif key == 'time_step':
            current['time_step'] = float(tokens[1])
            last_time_step = float(tokens[1])

        elif key == 'dump_thermo':
            current['dump_thermo'] = int(tokens[1])

        elif key == 'run':
            current['run'] = int(tokens[1])

            if 'time_step' not in current:
                if last_time_step is None:
                    raise ValueError("time_step 未定义且无法继承")
                current['time_step'] = last_time_step

            modules.append(current)
            current = {}

    return modules


# ============================================================
# 2. build time array
# ============================================================
def build_time_array(modules):

    time_list = []
    t_accum = 0.0  # fs

    for m in modules:
        if 'time_step' not in m:
            raise KeyError("模块缺少 time_step")
        if 'run' not in m:
            raise KeyError("模块缺少 run")

        dt = m['time_step']
        run = m['run']

        if 'dump_thermo' in m:
            dump = m['dump_thermo']

            n_points = run // dump
            dt_dump = dump * dt

            for k in range(1, n_points + 1):
                t = t_accum + k * dt_dump
                time_list.append(t)

        t_accum += run * dt

    return np.array(time_list) / 1000.0


# ============================================================
# 3. read natoms
# ============================================================
def read_natoms_from_model_xyz(filename):
    with open(filename, 'r') as f:
        first_line = f.readline().strip()
    return int(first_line)


# ============================================================
# main
# ============================================================
def main():

    if not os.path.isfile(runin_file):
        raise FileNotFoundError("run.in 不存在")

    modules = parse_run_in(runin_file)
    time = build_time_array(modules)

    if not os.path.isfile(thermo_file):
        raise FileNotFoundError("thermo.out 不存在")

    thermo = np.loadtxt(thermo_file)
    thermo = np.atleast_2d(thermo)

    T = thermo[:, 0]
    U_total = thermo[:, 2]

    # ===== model.xyz 可选 =====
    use_per_atom = False

    if os.path.isfile(model_file):
        try:
            natoms = read_natoms_from_model_xyz(model_file)
            U = U_total / natoms
            use_per_atom = True
            print(f"[INFO] Using per-atom energy: natoms = {natoms}")
        except Exception as e:
            print(f"[WARN] Failed to read model.xyz: {e}")
            U = U_total
    else:
        print("[INFO] model.xyz not found, using total energy")
        U = U_total

    # ===== other data =====
    Pxx = thermo[:, 3]
    Pyy = thermo[:, 4]
    Pzz = thermo[:, 5]
    Pyz = thermo[:, 6]
    Pxz = thermo[:, 7]
    Pxy = thermo[:, 8]

    ax = thermo[:, 9]
    ay = thermo[:, 10]
    az = thermo[:, 11]
    bx = thermo[:, 12]
    by = thermo[:, 13]
    bz = thermo[:, 14]
    cx = thermo[:, 15]
    cy = thermo[:, 16]
    cz = thermo[:, 17]

    a = np.sqrt(ax**2 + ay**2 + az**2)
    b = np.sqrt(bx**2 + by**2 + bz**2)
    c = np.sqrt(cx**2 + cy**2 + cz**2)

    # ===== length check =====
    if len(time) != len(T):
        print("WARNING:")
        print(f"time = {len(time)}, thermo = {len(T)}")

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

    # ===== Temperature =====
    plt.figure(figsize=(8, 6))
    plt.plot(time, T, '-', linewidth=lw)

    plt.xlabel(r'$t$ (ps)', fontsize=fs)
    plt.ylabel(r'$T$ (K)', fontsize=fs)

    plt.tick_params(axis='both', labelsize=fs-1)

    plt.tight_layout()
    plt.savefig('fig_temperature_vs_time.png', dpi=200)
    print('fig_temperature_vs_time.png saved ...')
    #plt.close()

    # ===== Energy =====
    plt.figure(figsize=(8, 6))
    plt.plot(time, U, '-', linewidth=lw)

    plt.xlabel(r'$t$ (ps)', fontsize=fs)

    if use_per_atom:
        plt.ylabel(r'$U$ (eV/atom)', fontsize=fs)
    else:
        plt.ylabel(r'$U$ (eV)', fontsize=fs)

    plt.tick_params(axis='both', labelsize=fs-1)

    plt.tight_layout()
    plt.savefig('fig_potential_energy_vs_time.png', dpi=200)
    print('fig_potential_energy_vs_time.png saved ...')
    #plt.close()

    # ===== Pressure =====
    plt.figure(figsize=(8, 6))
    plt.plot(time, Pxx, '-', linewidth=lw, label='Pxx')
    plt.plot(time, Pyy, '-.', linewidth=lw, label='Pyy')
    plt.plot(time, Pzz, '--', linewidth=lw, label='Pzz')

    plt.xlabel(r'$t$ (ps)', fontsize=fs)
    plt.ylabel(r'$P$ (GPa)', fontsize=fs)

    plt.tick_params(axis='both', labelsize=fs-1)

    plt.legend(fontsize=fs-4)
    plt.tight_layout()
    plt.savefig('fig_pressure_vs_time.png', dpi=200)
    print('fig_pressure_vs_time.png saved ...')
    #plt.close()

    # ===== Lattice =====
    plt.figure(figsize=(8, 6))
    plt.plot(time, a, '-', linewidth=lw, label='a')
    plt.plot(time, b, '-.', linewidth=lw, label='b')
    plt.plot(time, c, '--', linewidth=lw, label='c')

    plt.xlabel(r'$t$ (ps)', fontsize=fs)
    plt.ylabel(r'Lattice ($\mathrm{\AA}$)', fontsize=fs)

    plt.tick_params(axis='both', labelsize=fs-1)

    plt.legend(fontsize=fs-4)
    plt.tight_layout()
    plt.savefig('fig_lattice_vs_time.png', dpi=200)
    print('fig_lattice_vs_time.png saved ...')
    #plt.close()


# ============================================================
if __name__ == "__main__":
    main()
    
