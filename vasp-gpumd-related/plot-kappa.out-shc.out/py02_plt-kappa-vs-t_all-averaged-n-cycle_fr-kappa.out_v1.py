#!/usr/bin/env python3
#PS: The case directory name is regulated, which must be named by "job_x_m_n". x denotes a real number,
#    m denotes a integer(>=0) and n can be just 0, 1, 2, corresponding to x, y, z heat-flux direction.    

#PS: Necessary reaad file: run.in + kappa.out  

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# ===== parameters =====
fs = 22
lw = 2.2

r_dir = "./"
w_dir = "./"

# ===== 语义字典（可扩展）=====
name_dict = {
    'x': 'H-content-pct',
    'm': 'Sample',
    'n': 'Dir'
}


# ============================================================
# ===== 核心：解析 job_x_m_n =====
# ============================================================
def parse_job_name(dirname):
    """
    解析 job_x_m_n
    x: float >= 0
    m: int >= 1
    n: 0,1,2
    """

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


# ============================================================
# ===== 收集 & 分组 =====
# ============================================================
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
# ===== 解析 run.in
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
            if current and 'compute_hnemd' in current:
                modules.append(current)
            current = {}

        elif key == 'time_step':
            current['time_step'] = float(tokens[1])
            last_time_step = float(tokens[1])

        elif key == 'compute_hnemd':
            current['compute_hnemd'] = int(tokens[1])
            if float(tokens[2]) != 0:
                current["direction"] = "x"
            elif float(tokens[3]) != 0:
                current["direction"] = "y"
            elif float(tokens[4]) != 0:
                current["direction"] = "z"

        elif key == 'run':
            current['run'] = int(tokens[1])
            if 'time_step' not in current:
                current['time_step'] = last_time_step
            if 'compute_hnemd' in current:
                modules.append(current)
            current = {}

    return modules


def direction_to_n(direction):
    if direction == "x":
        return 0
    elif direction == "y":
        return 1
    elif direction == "z":
        return 2
    else:
        raise ValueError(f"未识别的 HNEMD 方向: {direction}")


def build_time_array(modules):
    time_list = []
    t_accum = 0.0

    for m in modules:
        dt = m['time_step']
        dump = m['compute_hnemd']
        run = m['run']

        n_points = run // dump
        dt_dump = dump * dt

        for k in range(1, n_points + 1):
            time_list.append(t_accum + k * dt_dump)

        t_accum += run * dt

    return np.array(time_list) / 1e6  # ns


def summary_kappa(modules, t, kappa_file):
    data = np.loadtxt(kappa_file)

    def ra(y, x):
        return cumulative_trapezoid(y, x, initial=0) / x

    kxi, kxo, kyi, kyo, kz = data.T

    kxi_ra = ra(kxi, t)
    kxo_ra = ra(kxo, t)
    kyi_ra = ra(kyi, t)
    kyo_ra = ra(kyo, t)
    kz_ra  = ra(kz, t)

    direction = modules[-1]['direction']

    if direction == "x":
        return kxi_ra + kxo_ra
    elif direction == "y":
        return kyi_ra + kyo_ra
    elif direction == "z":
        return kz_ra
    else:
        raise ValueError("未识别的 HNEMD 方向")


# ============================================================
# ===== 主程序 =====
# ============================================================
def main():

    groups = collect_jobs()

    for x_val, job_list in sorted(groups.items()):

        print(f"\nProcessing x = {x_val}")

        kappa_all = []
        time_ref = None

        for job, info in sorted(job_list):

            path = os.path.join(r_dir, job)

            runin = os.path.join(path, "run.in")
            kappa_file = os.path.join(path, "kappa.out")

            if not os.path.isfile(runin) or not os.path.isfile(kappa_file):
                print(f"WARNING: skip {job}")
                continue

            modules = parse_run_in(runin)

            # ===== 新增：检查目录名中的 n 是否和 run.in 中的 HNEMD 方向一致 =====
            n_val = info['n']
            runin_direction = modules[-1]['direction']
            runin_n = direction_to_n(runin_direction)

            if n_val != runin_n:
                print("************************************************************")
                print("WARNING: job directory direction index is inconsistent with run.in!")
                print(f"  job directory : {job}")
                print(f"  n from dirname: {n_val}")
                print(f"  direction from run.in: {runin_direction}  --> n = {runin_n}")
                print("  The script will continue anyway.")
                print("************************************************************")

            time = build_time_array(modules)
            kappa = summary_kappa(modules, time, kappa_file)

            n = min(len(time), len(kappa))
            time = time[:n]
            kappa = kappa[:n]

            if time_ref is None:
                time_ref = time
            else:
                if len(time_ref) != len(time) or not np.allclose(time_ref, time):
                    raise RuntimeError(f"time 不一致：{job}")

            kappa_all.append(kappa)

        kappa_all = np.array(kappa_all)

        if len(kappa_all) == 0:
            continue

        # ===== 平均曲线 =====
        kappa_mean_curve = np.mean(kappa_all, axis=0)

        # ===== 误差：先对每条独立曲线最后1/3取平均，再计算标准误差 =====
        tail_idx = len(time_ref) * 2 // 3
        tail_means = np.mean(kappa_all[:, tail_idx:], axis=1)

        mean = np.mean(tail_means)
        stderr = np.std(tail_means, ddof=1) / np.sqrt(len(tail_means))

        print(f"  kappa = {mean:.4f} +/- {stderr:.4f}")

        # ===== 绘图 =====
        plt.figure(figsize=(8, 6))

        for kappa in kappa_all:
            plt.plot(time_ref, kappa, '-', color='blue', alpha=0.2, linewidth=lw-1)

        plt.plot(time_ref, kappa_mean_curve, 'b--', linewidth=lw,
                 label=rf'$\kappa = {mean:.2f} \pm {stderr:.2f}$')

        plt.xlabel(r'$t$ (ns)', fontsize=fs)
        plt.ylabel(r'$\kappa$ (Wm$^{-1}$K$^{-1}$)', fontsize=fs)
        plt.gca().tick_params(axis='both', labelsize=fs-3)

        plt.title(rf'${name_dict["x"]} = {x_val}$', fontsize=fs)

        plt.legend(fontsize=fs)
        plt.tight_layout()

        fname = os.path.join(
            w_dir,
            f'fig_kappa-time_collected-average_{name_dict["x"]}-{x_val}.png'
        )

        plt.savefig(fname, dpi=200)
        plt.close()

        print(f"Saved: {fname}")


if __name__ == "__main__":
    main()