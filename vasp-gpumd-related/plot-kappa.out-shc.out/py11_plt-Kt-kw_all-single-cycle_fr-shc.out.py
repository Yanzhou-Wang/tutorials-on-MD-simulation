#!/usr/bin/env python3
# PS: The case directory name is regulated, which must be named by "job_x_m_n".
#     x denotes a real number, m denotes an integer (>=1), and n can be only
#     0, 1, 2, corresponding to x, y, z heat-flux direction.
#
# PS: Necessary read files: run.in + shc.out + thermo.out

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# ============================================================
# ===== parameters =====
# ============================================================
fs = 22
lw = 2.2

r_dir = "./"
w_dir = "./"

# ===== semantic dictionary =====
name_dict = {
    'x': 'H-cont-pct',
    'm': 'Sap',
    'n': 'Dir'
}


# ============================================================
# ===== parse job_x_m_n =====
# ============================================================
def parse_job_name(dirname):
    if not dirname.startswith("job_"):
        return None

    parts = dirname.split('_')

    if len(parts) != 4:
        raise RuntimeError(f"目录名格式错误（必须为 job_x_m_n）：{dirname}")

    _, x_str, m_str, n_str = parts

    try:
        x = float(x_str)
    except:
        raise RuntimeError(f"x 不是合法实数：{dirname}")

    if x < 0:
        raise RuntimeError(f"x 必须 >= 0：{dirname}")

    if not m_str.isdigit():
        raise RuntimeError(f"m 必须是正整数：{dirname}")

    m = int(m_str)
    if m <= 0:
        raise RuntimeError(f"m 必须 >= 1：{dirname}")

    if not n_str.isdigit():
        raise RuntimeError(f"n 必须是整数（0/1/2）：{dirname}")

    n = int(n_str)
    if n not in [0, 1, 2]:
        raise RuntimeError(f"n 必须是 0/1/2：{dirname}")

    return {'x': x, 'm': m, 'n': n}


def collect_jobs():
    jobs = []

    for d in os.listdir(r_dir):
        info = parse_job_name(d)

        if info is None:
            continue

        jobs.append((d, info))

    return jobs


# ============================================================
# ===== parse run.in =====
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

        elif key == 'compute_hnemd':
            if len(tokens) < 5:
                raise RuntimeError(
                    "compute_hnemd 参数不足，应为: "
                    "compute_hnemd <output_interval> <Fe_x> <Fe_y> <Fe_z>"
                )

            current['compute_hnemd'] = tokens[1:]
            current['hnemd_output_interval'] = int(tokens[1])

            Fe_x = float(tokens[2])
            Fe_y = float(tokens[3])
            Fe_z = float(tokens[4])

            current['Fe_components'] = [Fe_x, Fe_y, Fe_z]

            fe_tol = 1.0e-15
            nonzero_fe = [
                (i, val) for i, val in enumerate([Fe_x, Fe_y, Fe_z])
                if abs(val) > fe_tol
            ]

            if len(nonzero_fe) == 1:
                current['Fe_direction'] = nonzero_fe[0][0]
                current['Fe'] = nonzero_fe[0][1]
            else:
                current['Fe_direction'] = None
                current['Fe'] = None

        elif key == 'compute_shc':
            if len(tokens) < 6:
                raise RuntimeError(
                    "compute_shc 参数不足，应为: "
                    "compute_shc <sample_interval> <Nc> <transport_direction> <num_omega> <max_omega>"
                )

            current['compute_shc'] = tokens[1:]
            current['sample_interval'] = int(tokens[1])
            current['Nc'] = int(tokens[2])
            current['transport_direction'] = int(tokens[3])
            current['num_omega'] = int(tokens[4])
            current['max_omega'] = float(tokens[5])

        elif key == 'run':
            current['run'] = int(tokens[1])

            if 'time_step' not in current:
                if last_time_step is not None:
                    current['time_step'] = last_time_step

            modules.append(current)
            current = {}

    return modules


def get_shc_module(modules, job):
    shc_modules = [m for m in modules if 'compute_shc' in m]

    if len(shc_modules) == 0:
        raise RuntimeError(f"{job}: run.in 中没有找到 compute_shc 模块")

    if len(shc_modules) > 1:
        print("************************************************************")
        print("WARNING: More than one compute_shc module found in run.in!")
        print(f"  job directory : {job}")
        print("  The last compute_shc module will be used.")
        print("************************************************************")

    return shc_modules[-1]


def get_hnemd_module(modules, shc_module, job):
    if 'compute_hnemd' in shc_module:
        return shc_module

    print("************************************************************")
    print("WARNING: compute_hnemd and compute_shc are not in the same module!")
    print(f"  job directory : {job}")
    print("  The script will try to use the last compute_hnemd module in run.in.")
    print("************************************************************")

    hnemd_modules = [m for m in modules if 'compute_hnemd' in m]

    if len(hnemd_modules) == 0:
        raise RuntimeError(f"{job}: run.in 中没有找到 compute_hnemd，无法获得 Fe")

    return hnemd_modules[-1]


def parse_temperature_from_ensemble(module, job):
    if 'ensemble' not in module:
        raise RuntimeError(f"{job}: compute_shc 模块中没有找到 ensemble，无法读取温度 T")

    ens = module['ensemble']

    if len(ens) != 4:
        print("************************************************************")
        print("WARNING: ensemble line may not be a standard NVT format!")
        print(f"  job directory : {job}")
        print(f"  ensemble      : {' '.join(ens)}")
        print("  Expected format: ensemble nvt_xxx <T_1> <T_2> <T_coup>")
        print("  The script will continue anyway.")
        print("************************************************************")

    ensemble_name = ens[0]

    if 'nvt' not in ensemble_name:
        print("************************************************************")
        print("WARNING: ensemble in compute_shc module does not seem to be NVT!")
        print(f"  job directory : {job}")
        print(f"  ensemble      : {' '.join(ens)}")
        print("  The script will continue anyway.")
        print("************************************************************")

    try:
        T1 = float(ens[1])
        T2 = float(ens[2])
    except:
        raise RuntimeError(f"{job}: 无法从 ensemble 行解析 T_1 和 T_2")

    if T1 != T2:
        print("************************************************************")
        print("WARNING: T_1 and T_2 in ensemble are different!")
        print(f"  job directory : {job}")
        print(f"  T_1 = {T1}")
        print(f"  T_2 = {T2}")
        print("  T_1 will be used as the temperature T.")
        print("************************************************************")

    return T1


def parse_Fe_from_hnemd_module(module, job):
    if 'compute_hnemd' not in module:
        raise RuntimeError(f"{job}: 给定模块中没有 compute_hnemd，无法读取 Fe")

    Fe_components = module.get('Fe_components', None)

    if Fe_components is None:
        raise RuntimeError(f"{job}: 无法解析 compute_hnemd 中的 Fe_x, Fe_y, Fe_z")

    fe_tol = 1.0e-15
    nonzero_fe = [
        (i, val) for i, val in enumerate(Fe_components)
        if abs(val) > fe_tol
    ]

    if len(nonzero_fe) != 1:
        raise RuntimeError(
            f"{job}: compute_hnemd 中应当只有一个非零 Fe 分量，"
            f"当前 Fe_x, Fe_y, Fe_z = {Fe_components}"
        )

    Fe_direction = nonzero_fe[0][0]
    Fe = nonzero_fe[0][1]

    if Fe < 0:
        print("************************************************************")
        print("WARNING: Fe should be non-negative!")
        print(f"  job directory : {job}")
        print(f"  Fe = {Fe}")
        print("  The script will continue anyway.")
        print("************************************************************")

    return Fe, Fe_direction


def check_runin_consistency(job, info, shc_module, hnemd_module):
    n_val = info['n']
    shc_n = shc_module['transport_direction']

    if n_val != shc_n:
        print("************************************************************")
        print("WARNING: job directory direction index is inconsistent with compute_shc!")
        print(f"  job directory        : {job}")
        print(f"  n from dirname       : {n_val}")
        print(f"  direction from shc   : {shc_n}")
        print("  The script will continue anyway.")
        print("************************************************************")

    if 'Fe_components' in hnemd_module:
        fe_tol = 1.0e-15
        Fe_components = hnemd_module['Fe_components']

        nonzero_fe = [
            (i, val) for i, val in enumerate(Fe_components)
            if abs(val) > fe_tol
        ]

        if len(nonzero_fe) != 1:
            print("************************************************************")
            print("WARNING: invalid compute_hnemd driving force setting!")
            print(f"  job directory : {job}")
            print(f"  Fe_x, Fe_y, Fe_z = {Fe_components}")
            print("  Expected exactly one non-zero component.")
            print("  The script will continue anyway.")
            print("************************************************************")
        else:
            hnemd_n = nonzero_fe[0][0]

            if hnemd_n != shc_n:
                print("************************************************************")
                print("WARNING: compute_hnemd direction is inconsistent with compute_shc direction!")
                print(f"  job directory        : {job}")
                print(f"  direction from hnemd : {hnemd_n}")
                print(f"  direction from shc   : {shc_n}")
                print("  The script will continue anyway.")
                print("************************************************************")


# ============================================================
# ===== read thermo.out and get Lx, Ly, Lz =====
# ============================================================
def read_box_from_thermo(thermo_file, job):
    axis_tol = 1.0e-6

    thermo = np.loadtxt(thermo_file)

    if thermo.ndim == 1:
        last = thermo
    else:
        last = thermo[-1, :]

    if len(last) < 18:
        raise RuntimeError(f"{thermo_file} 列数不足，无法读取 box matrix")

    ax, ay, az = last[9:12]
    bx, by, bz = last[12:15]
    cx, cy, cz = last[15:18]

    if abs(ay) > axis_tol or abs(az) > axis_tol:
        print("************************************************************")
        print("WARNING: a-vector is not parallel to x direction!")
        print(f"  job directory : {job}")
        print(f"  a = ({ax:.6e}, {ay:.6e}, {az:.6e})")
        print("  The script will continue anyway.")
        print("************************************************************")

    if abs(bx) > axis_tol or abs(bz) > axis_tol:
        print("************************************************************")
        print("WARNING: b-vector is not parallel to y direction!")
        print(f"  job directory : {job}")
        print(f"  b = ({bx:.6e}, {by:.6e}, {bz:.6e})")
        print("  The script will continue anyway.")
        print("************************************************************")

    if abs(cx) > axis_tol or abs(cy) > axis_tol:
        print("************************************************************")
        print("WARNING: c-vector is not parallel to z direction!")
        print(f"  job directory : {job}")
        print(f"  c = ({cx:.6e}, {cy:.6e}, {cz:.6e})")
        print("  The script will continue anyway.")
        print("************************************************************")

    Lx = abs(ax)
    Ly = abs(by)
    Lz = abs(cz)

    return Lx, Ly, Lz


def get_length_along_direction(n_val, Lx, Ly, Lz):
    if n_val == 0:
        return Lx
    elif n_val == 1:
        return Ly
    elif n_val == 2:
        return Lz
    else:
        raise ValueError("n must be 0, 1, or 2")


# ============================================================
# ===== read and process shc.out =====
# ============================================================
def read_shc_file(shc_file, Nc, num_omega):
    shc = np.loadtxt(shc_file)

    Nt = 2 * Nc - 1
    n_required = Nt + num_omega

    if shc.shape[0] != n_required:
        raise RuntimeError(
            f"{shc_file} 行数错误：需要 {n_required} 行 "
            f"(2*Nc-1+num_omega = 2*{Nc}-1+{num_omega})，实际 {shc.shape[0]} 行"
        )

    shc_Kt = shc[:Nt, :]
    shc_kw = shc[Nt:, :]

    return shc_Kt, shc_kw


def calc_Kt(shc_Kt, tran_length):
    time_in_ps = shc_Kt[:, 0]
    Kt = np.sum(shc_Kt[:, 1:3], axis=1) / tran_length
    return time_in_ps, Kt


def calc_spectral_kappa(shc_kw, Fe, T, Lx, Ly, Lz):
    omega = shc_kw[:, 0]
    nu = omega / (2.0 * np.pi)

    Jw = np.sum(shc_kw[:, 1:3], axis=1)

    V = Lx * Ly * Lz
    kw = Jw * 1602.17662 / (Fe * T * V)

    kappa_integrated = trapezoid(kw, nu)

    return nu, kw, kappa_integrated


# ============================================================
# ===== plotting functions =====
# ============================================================
def make_title(info):
    x_val = info['x']
    m_val = info['m']
    n_val = info['n']

    return rf'${name_dict["x"]} = {x_val},\ {name_dict["m"]} = {m_val},\ {name_dict["n"]} = {n_val}$'


def make_file_tag(info):
    x_val = info['x']
    m_val = info['m']
    n_val = info['n']

    return f'{name_dict["x"]}-{x_val}_{name_dict["m"]}-{m_val}_{name_dict["n"]}-{n_val}'


def plot_Kt(time_in_ps, Kt, info):
    plt.figure(figsize=(8, 6))

    plt.plot(time_in_ps, Kt, 'b-', linewidth=lw)

    plt.xlabel(r'$t$ (ps)', fontsize=fs)
    plt.ylabel(r'$K(t)$ (eV/ps)', fontsize=fs)
    plt.gca().tick_params(axis='both', labelsize=fs-3)

    plt.title(make_title(info), fontsize=fs)

    plt.tight_layout()

    fname = os.path.join(
        w_dir,
        f'fig_Kt_{make_file_tag(info)}.png'
    )

    plt.savefig(fname, dpi=200)
    plt.close()

    return fname


def plot_kw(nu, kw, kappa_integrated, info):
    plt.figure(figsize=(8, 6))

    plt.plot(nu, kw, 'b-', linewidth=lw,
             label=rf'$\kappa = {kappa_integrated:.2f}$')

    plt.xlabel(r'$\omega/2\pi$ (THz)', fontsize=fs)
    plt.ylabel(r'$\kappa(\omega)$ (Wm$^{-1}$K$^{-1}$THz$^{-1}$)', fontsize=fs)
    plt.gca().tick_params(axis='both', labelsize=fs-3)

    plt.title(make_title(info), fontsize=fs)

    plt.legend(fontsize=fs)
    plt.tight_layout()

    fname = os.path.join(
        w_dir,
        f'fig_kw_{make_file_tag(info)}.png'
    )

    plt.savefig(fname, dpi=200)
    plt.close()

    return fname


# ============================================================
# ===== main =====
# ============================================================
def main():

    jobs = collect_jobs()

    for job, info in sorted(jobs):

        print(f"\nProcessing {job}")

        path = os.path.join(r_dir, job)

        runin = os.path.join(path, "run.in")
        shc_file = os.path.join(path, "shc.out")
        thermo_file = os.path.join(path, "thermo.out")

        if not os.path.isfile(runin) or not os.path.isfile(shc_file) or not os.path.isfile(thermo_file):
            print(f"WARNING: skip {job}, run.in or shc.out or thermo.out missing")
            continue

        modules = parse_run_in(runin)
        shc_module = get_shc_module(modules, job)
        hnemd_module = get_hnemd_module(modules, shc_module, job)

        check_runin_consistency(job, info, shc_module, hnemd_module)

        Nc = shc_module['Nc']
        num_omega = shc_module['num_omega']
        max_omega = shc_module['max_omega']
        shc_direction = shc_module['transport_direction']

        T = parse_temperature_from_ensemble(shc_module, job)
        Fe, Fe_direction = parse_Fe_from_hnemd_module(hnemd_module, job)

        Lx, Ly, Lz = read_box_from_thermo(thermo_file, job)

        tran_length = get_length_along_direction(shc_direction, Lx, Ly, Lz)

        print(f"  Nc = {Nc}")
        print(f"  num_omega = {num_omega}")
        print(f"  max_omega = {max_omega:.6f} THz")
        print(f"  compute_shc direction = {shc_direction}")
        print(f"  compute_hnemd direction = {Fe_direction}")
        print(f"  T = {T:.6f} K")
        print(f"  Fe = {Fe:.6e}")
        print(f"  Lx = {Lx:.6f} Angstrom")
        print(f"  Ly = {Ly:.6f} Angstrom")
        print(f"  Lz = {Lz:.6f} Angstrom")
        print(f"  length used for K(t) = {tran_length:.6f} Angstrom")

        shc_Kt, shc_kw = read_shc_file(shc_file, Nc, num_omega)

        time_in_ps, Kt = calc_Kt(shc_Kt, tran_length)
        nu, kw, kappa_integrated = calc_spectral_kappa(shc_kw, Fe, T, Lx, Ly, Lz)

        print(f"  integrated kappa = {kappa_integrated:.4f} W/m/K")

        fname_Kt = plot_Kt(time_in_ps, Kt, info)
        fname_kw = plot_kw(nu, kw, kappa_integrated, info)

        print(f"Saved: {fname_Kt}")
        print(f"Saved: {fname_kw}")


if __name__ == "__main__":
    main()