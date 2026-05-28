#!/usr/bin/env python3
# PS: The case directory name is regulated, which must be named by "job_x_m_n".
#     x denotes a real number, m denotes an integer (>=1), and n can be only
#     0, 1, 2, corresponding to x, y, z heat-flux direction.
#
# PS: Necessary read files: run.in + shc.out + thermo.out
# PS: This script groups all jobs with the same x and plots averaged classical kw only.

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
    groups = {}

    for d in os.listdir(r_dir):
        info = parse_job_name(d)

        if info is None:
            continue

        x = info['x']
        groups.setdefault(x, []).append((d, info))

    return groups


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

    Fe = nonzero_fe[0][1]

    if Fe < 0:
        print("************************************************************")
        print("WARNING: Fe should be non-negative!")
        print(f"  job directory : {job}")
        print(f"  Fe = {Fe}")
        print("  The script will continue anyway.")
        print("************************************************************")

    return Fe


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

    fe_tol = 1.0e-15
    Fe_components = hnemd_module.get('Fe_components', None)

    if Fe_components is None:
        print("************************************************************")
        print("WARNING: failed to parse compute_hnemd driving force setting!")
        print(f"  job directory : {job}")
        print("  The script will continue anyway.")
        print("************************************************************")
        return

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

    shc_kw = shc[Nt:, :]

    return shc_kw


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
def make_title_x(x_val):
    return rf'${name_dict["x"]} = {x_val}$'


def make_file_tag_x(x_val):
    return f'{name_dict["x"]}-{x_val}'


def plot_grouped_kw(x_val, nu_ref, kw_all, kappa_all):
    kw_mean = np.mean(kw_all, axis=0)
    kappa_mean = np.mean(kappa_all)

    if len(kappa_all) > 1:
        kappa_stderr = np.std(kappa_all, ddof=1) / np.sqrt(len(kappa_all))
    else:
        kappa_stderr = 0.0

    plt.figure(figsize=(8, 6))

    for kw in kw_all:
        plt.plot(nu_ref, kw, '-', color='blue', alpha=0.2, linewidth=lw-1)

    plt.plot(
        nu_ref, kw_mean, 'b--', linewidth=lw,
        label=rf'$\kappa = {kappa_mean:.2f} \pm {kappa_stderr:.2f}$'
    )

    plt.xlabel(r'$\omega/2\pi$ (THz)', fontsize=fs)
    plt.ylabel(r'$\kappa(\omega)$ (Wm$^{-1}$K$^{-1}$THz$^{-1}$)', fontsize=fs)
    plt.gca().tick_params(axis='both', labelsize=fs-3)

    plt.title(make_title_x(x_val), fontsize=fs)

    plt.legend(fontsize=fs)
    plt.tight_layout()

    fname = os.path.join(
        w_dir,
        f'fig_kw_collected-average_{make_file_tag_x(x_val)}.png'
    )

    plt.savefig(fname, dpi=200)
    plt.close()

    return fname, kappa_mean, kappa_stderr


# ============================================================
# ===== main =====
# ============================================================
def main():

    groups = collect_jobs()

    for x_val, job_list in sorted(groups.items()):

        print(f"\nProcessing x = {x_val}")

        nu_ref = None
        kw_all = []
        kappa_all = []

        for job, info in sorted(job_list):

            print(f"  Processing {job}")

            path = os.path.join(r_dir, job)

            runin = os.path.join(path, "run.in")
            shc_file = os.path.join(path, "shc.out")
            thermo_file = os.path.join(path, "thermo.out")

            if not os.path.isfile(runin) or not os.path.isfile(shc_file) or not os.path.isfile(thermo_file):
                print(f"  WARNING: skip {job}, run.in or shc.out or thermo.out missing")
                continue

            modules = parse_run_in(runin)
            shc_module = get_shc_module(modules, job)
            hnemd_module = get_hnemd_module(modules, shc_module, job)

            check_runin_consistency(job, info, shc_module, hnemd_module)

            Nc = shc_module['Nc']
            num_omega = shc_module['num_omega']

            T = parse_temperature_from_ensemble(shc_module, job)
            Fe = parse_Fe_from_hnemd_module(hnemd_module, job)

            Lx, Ly, Lz = read_box_from_thermo(thermo_file, job)

            shc_kw = read_shc_file(shc_file, Nc, num_omega)

            nu, kw, kappa_integrated = calc_spectral_kappa(shc_kw, Fe, T, Lx, Ly, Lz)

            if nu_ref is None:
                nu_ref = nu
            else:
                if len(nu_ref) != len(nu) or not np.allclose(nu_ref, nu):
                    raise RuntimeError(f"{job}: nu frequency grid is inconsistent with other jobs for x = {x_val}")

            kw_all.append(kw)
            kappa_all.append(kappa_integrated)

            print(f"    kappa = {kappa_integrated:.4f} W/m/K")

        if len(kw_all) == 0:
            print(f"WARNING: no valid jobs found for x = {x_val}")
            continue

        kw_all = np.array(kw_all)
        kappa_all = np.array(kappa_all)

        fname, k_mean, k_se = plot_grouped_kw(
            x_val, nu_ref, kw_all, kappa_all
        )

        print(f"  averaged kappa = {k_mean:.4f} +/- {k_se:.4f} W/m/K")
        print(f"Saved: {fname}")


if __name__ == "__main__":
    main()
    
    