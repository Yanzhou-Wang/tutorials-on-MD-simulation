#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
//////////////// yanzhowang@gmail.com //////////////////////////
Extract forces and stress components from vasprun.xml and plot:
  Fig.1: Atomic forces (Fmax, Frms) per ionic step (eV/Å)
  Fig.2: Stress tensor components per ionic step (Voigt: xx, yy, zz, yz, xz, xy) in eV/Å^3
  Fig.3: Generalized cell forces (Gxx, Gyy, Gzz, Gyz, Gxz, Gxy) derived from stress×area (eV/Å)

Outputs:
  - fig_atomic-force-vs-step.png
  - fig_cell-stress-vs-step.png
  - fig_generalized-cell-force-vs-step.png
  (optional text files such as forces_list.txt, stress_components.txt, gforces_components.txt
   can be enabled in the commented-out sections)
"""


import os
import re
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read  # ASE 自动分发到 read_vasp_xml


# === 用户可调部分 =======================================================
# 主目录（相对当前脚本运行位置）
main_path = "./"

# 用户需要处理的算例目录（这个要用于与 main_path 的拼接）
job_items = [
    "job_xxx_0",
    "job_yyy_0",
]

# 用户指定输入结构文件名
in_file = "vasprun.xml"

# 用户指定的图片保存路径
w_dir = "./"

# 绘图字体大小
fs = 16
# =======================================================================


def parse_job_name(job_name):
    """
    解析 job 目录名。

    目录命名规则固定为：
        job_xxx_0

    即：
        第1段：以 job 开头
        第2段：字符串，例如 xxx
        第3段：大于等于0的整数

    返回：
        jn_p2, jn_p3
    """
    parts = job_name.split("_")

    if len(parts) != 3:
        raise ValueError(
            f"Invalid job directory name: {job_name}. "
            "Expected format: job_xxx_0"
        )

    jn_p1, jn_p2, jn_p3_str = parts

    if not jn_p1.startswith("job"):
        raise ValueError(
            f"Invalid job directory name: {job_name}. "
            "The first field should start with 'job'."
        )

    try:
        jn_p3 = int(jn_p3_str)
    except ValueError:
        raise ValueError(
            f"Invalid job directory name: {job_name}. "
            "The third field should be an integer >= 0."
        )

    if jn_p3 < 0:
        raise ValueError(
            f"Invalid job directory name: {job_name}. "
            "The third field should be an integer >= 0."
        )

    return jn_p2, jn_p3


# ---- 最简 & 稳妥：仅扫描 <incar> 块，正则取 EDIFFG ----
def get_EDIFFG_from_incar_block(filename: str):
    in_incar = False
    buf = []
    with open(filename, "r", encoding="ISO-8859-1", errors="ignore") as f:
        for line in f:
            if "<incar" in line:
                in_incar = True
            if in_incar:
                buf.append(line)
            if in_incar and "</incar>" in line:
                break
    block = "".join(buf)
    m = re.search(r'name\s*=\s*"?EDIFFG"?\s*>\s*([-\+0-9Ee\.]+)\s*<', block, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        val = float(m.group(1))
        print(f"EDIFFG found (text-scan <incar>): {val}")
        return val
    except Exception:
        return None

def parse_vasprun_with_ase(filename: str):
    atoms_list = read(filename, index=":")  # list[Atoms]
    steps, Fmax, Frms = [], [], []
    stress_components, forces_list = [], []
    face_areas = []  # 新增：每步的 (A_x, A_y, A_z)

    for i, at in enumerate(atoms_list):
        steps.append(i + 1)

        # forces (eV/Å)
        forces = at.get_forces()
        norms = np.linalg.norm(forces, axis=1)
        Fmax.append(np.max(norms))
        Frms.append(np.sqrt(np.mean(norms**2)))
        forces_list.append(forces)

        # stress (Voigt: xx, yy, zz, yz, xz, xy) eV/Å^3
        s = at.get_stress(apply_constraint=False)
        if s is None:
            s = np.full(6, np.nan, dtype=float)
        stress_components.append(s)

        # ---- 新增：计算三个面的面积 (Å^2)
        a, b, c = at.get_cell()  # 3×3
        Ax = np.linalg.norm(np.cross(b, c))  # 面 yz，法向 x
        Ay = np.linalg.norm(np.cross(c, a))  # 面 xz，法向 y
        Az = np.linalg.norm(np.cross(a, b))  # 面 xy，法向 z
        face_areas.append([Ax, Ay, Az])

    return (np.array(steps, dtype=int),
            np.array(Fmax, dtype=float),
            np.array(Frms, dtype=float),
            np.vstack(stress_components).astype(float),
            forces_list,
            np.array(face_areas, dtype=float))  # ← 新增返回值

'''
def save_forces_and_stresses(forces_list, stress_components):
    # forces
    with open("forces_list.txt", "w") as f:
        for istep, step_forces in enumerate(forces_list, start=1):
            f.write(f"# step {istep}\n")
            for fx, fy, fz in step_forces:
                f.write(f"{fx:.8f} {fy:.8f} {fz:.8f}\n")
    print("Saved: forces_list.txt")

    # stress (Voigt: xx yy zz yz xz xy)
    with open("stress_components.txt", "w") as f:
        for istep, s in enumerate(stress_components, start=1):
            f.write(f"# step {istep}\n")
            f.write("{:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n".format(
                s[0], s[1], s[2], s[3], s[4], s[5]))
    print("Saved: stress_components.txt")
'''

# -----------------------------
def plot_forces(steps, Fmax, Frms, EDIFFG=None, fig_prefix="", w_dir="./"):
    plt.figure(figsize=(8, 6))
    plt.plot(steps, Fmax, marker="o", markerfacecolor="None", label="Fmax (eV/Å)")
    plt.plot(steps, Frms, marker="s", markerfacecolor="None", label="Frms (eV/Å)")
    if EDIFFG is not None:
        plt.axhline(abs(EDIFFG), linestyle="--", color="black", label=f"|EDIFFG|={abs(EDIFFG)}")
    plt.xlabel("Ionic step", fontsize=fs)
    plt.ylabel("Force (eV/Å)", fontsize=fs)
    plt.title("Atomic force vs. ionic step", fontsize=fs)
    plt.grid(True, linestyle=":")
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.legend(fontsize=fs-2)
    plt.tight_layout()

    fig_name = f"fig_{fig_prefix}_atomic-force-vs-step.png"
    fig_path = os.path.join(w_dir, fig_name)
    plt.savefig(fig_path, dpi=200)
#    plt.show()  # 让 Spyder Plots 捕获显示
    print(fig_path)

def plot_stresses(steps, stress_components, fig_prefix="", w_dir="./"):
#    labels = ["σ_xx", "σ_yy", "σ_zz", "σ_yz", "σ_xz", "σ_xy"]  # ASE 返回顺序
    labels = [r"$\sigma_{xx}$", r"$\sigma_{yy}$", r"$\sigma_{zz}$",
          r"$\sigma_{yz}$", r"$\sigma_{xz}$", r"$\sigma_{xy}$"]
    plt.figure(figsize=(8, 6))
    for i in range(6):
        plt.plot(steps, stress_components[:, i], marker="o", markerfacecolor="None", label=labels[i])
    plt.xlabel("Ionic step", fontsize=fs)
    plt.ylabel("Stress (eV/Å³)", fontsize=fs)
    plt.title("Cell stress vs. ionic step", fontsize=fs)
    plt.grid(True, linestyle=":")
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.legend(ncol=2, fontsize=fs-2)
    plt.tight_layout()

    fig_name = f"fig_{fig_prefix}_cell-stress-vs-step.png"
    fig_path = os.path.join(w_dir, fig_name)
    plt.savefig(fig_path, dpi=200)
#    plt.show()  # 让 Spyder Plots 捕获显示
    print("Saved:", fig_path)

def plot_gforces(steps, stress_components, face_areas, EDIFFG=None, fig_prefix="", w_dir="./"):
    """
    根据应力 (eV/Å^3) 和对应面的面积 (Å^2) 计算广义力 (eV/Å)，并绘图。
    Voigt: [xx, yy, zz, yz, xz, xy]
    映射：xx/yz 用 A_x；yy/xz 用 A_y；zz/xy 用 A_z
    """
    # 面面积
    Ax = face_areas[:, 0]
    Ay = face_areas[:, 1]
    Az = face_areas[:, 2]

    # 逐分量计算广义力 (eV/Å)
    G = np.empty_like(stress_components)
    G[:, 0] = stress_components[:, 0] * Ax  # xx
    G[:, 1] = stress_components[:, 1] * Ay  # yy
    G[:, 2] = stress_components[:, 2] * Az  # zz
    G[:, 3] = stress_components[:, 3] * Ax  # yz -> 面法向 x
    G[:, 4] = stress_components[:, 4] * Ay  # xz -> 面法向 y
    G[:, 5] = stress_components[:, 5] * Az  # xy -> 面法向 z

    # 绘图
#    labels = ["G_xx", "G_yy", "G_zz", "G_yz", "G_xz", "G_xy"]
    labels = [r"$G_{xx}$", r"$G_{yy}$", r"$G_{zz}$",
          r"$G_{yz}$", r"$G_{xz}$", r"$G_{xy}$"]

    plt.figure(figsize=(8, 6))
    for i in range(6):
        plt.plot(steps, G[:, i], marker="o",  markerfacecolor="None", label=labels[i])
    if EDIFFG is not None:
        plt.axhline(abs(EDIFFG), linestyle="--", color="black", label=f"|EDIFFG|={abs(EDIFFG)}")
        plt.axhline(-abs(EDIFFG), linestyle="--", color="black")  # 对称参考，方便观察正负
    plt.xlabel("Ionic step", fontsize=fs)
    plt.ylabel("Generalized force (eV/Å)", fontsize=fs)
    plt.title("Generalized cell force vs. ionic step", fontsize=fs)
    plt.grid(True, linestyle=":")
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.legend(ncol=2, fontsize=fs-2)
    plt.tight_layout()

    fig_name = f"fig_{fig_prefix}_generalized-cell-force-vs-step.png"
    fig_path = os.path.join(w_dir, fig_name)
    plt.savefig(fig_path, dpi=200)
#    plt.show()
    print("Saved:", fig_path)


'''
    # （可选）保存到文本，便于核查
    with open("gforces_components.txt", "w") as f:
        for istep in range(len(steps)):
            f.write(f"# step {steps[istep]}\n")
            f.write("{:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n".format(
                G[istep, 0], G[istep, 1], G[istep, 2], G[istep, 3], G[istep, 4], G[istep, 5]))
    print("Saved: gforces_components.txt")
'''

def main():
    os.makedirs(w_dir, exist_ok=True)

    for job_name in job_items:
        try:
            jn_p2, jn_p3 = parse_job_name(job_name)
            fig_prefix = f"{jn_p2}-{jn_p3}"

            filename = os.path.join(main_path, job_name, in_file)

            print(f"\nProcessing job: {job_name}")
            print(f"Input file    : {filename}")
            print(f"Figure prefix : {fig_prefix}")

            if not os.path.exists(filename):
                print(f"Error: {filename} not found.")
                continue

            # 1) 读取 EDIFFG
            EDIFFG = get_EDIFFG_from_incar_block(filename)
            if EDIFFG is None:
                print("Warning: EDIFFG not found in <incar> block, using preset -1e-2.")
                EDIFFG = -1e-2
            if EDIFFG > 0:
                    print("======== WARNING: EDIFFG is a positive value, which is not suitable for "
                          "relaxation of atomic positions. I hope you know what you are doing. =====")
            # 2) 解析 forces / stress（注意多接一个 face_areas）
            steps, Fmax, Frms, stress6, forces_list, face_areas = parse_vasprun_with_ase(filename)

            # 3) 导出
        #    save_forces_and_stresses(forces_list, stress6)


            # 4) 分开绘图：两张图 + 广义力
            plot_forces(steps, Fmax, Frms, EDIFFG, fig_prefix=fig_prefix, w_dir=w_dir)
            plot_stresses(steps, stress6, fig_prefix=fig_prefix, w_dir=w_dir)
            plot_gforces(steps, stress6, face_areas, EDIFFG, fig_prefix=fig_prefix, w_dir=w_dir)  # ← 新增

        except Exception as e:
            print(f"[ERROR] Failed to process job: {job_name}")
            print(f"        Reason: {e}")


if __name__ == "__main__":
    main()