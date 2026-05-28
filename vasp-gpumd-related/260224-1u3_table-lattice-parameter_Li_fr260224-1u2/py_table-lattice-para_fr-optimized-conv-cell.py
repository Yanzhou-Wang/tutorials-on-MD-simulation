#!/usr/bin/env python3
import os
import numpy as np

# === 用户可调部分 =======================================================
# 主目录（相对当前脚本运行位置）
main_path = "../260224-1u2_identify-prim-conv-orth_Li_fr260224-1"

# 用户需要处理的算例目录（这个要用于与main_path的拼接）
job_items = [
    "id-mp-604313_Li",
]

# 用户指定输入结构文件名（一定要是晶胞！！！）
in_file = "conv.vasp" 

# 用户指定输出文件名
out_file = "result-lattice-para.txt"
# =======================================================================

def angle_between(u, v):
    """返回向量 u, v 之间的夹角（单位：度）"""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return float("nan")
    cosang = np.dot(u, v) / (nu * nv)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def read_lattice_from_conv_vasp(path):
    """
    从 VASP 格式的 POSCAR/CONTCAR/conv.vasp 读取晶胞基矢。
    假定：
        第 1 行：注释
        第 2 行：全局缩放因子（s）
        第 3-5 行：a, b, c 三个基矢 (以 s 为缩放)
    返回：a, b, c（三个 3D 向量）
    """
    with open(path, "r") as f:
        lines = f.readlines()

    if len(lines) < 5:
        raise ValueError(f"{path} 格式不完整，行数不足 5 行。")

    # 缩放因子（可能是 1.0）
    scale = float(lines[1].split()[0])

    # 第 3,4,5 行为基矢
    a_vec = np.array([float(x) for x in lines[2].split()[:3]], dtype=float) * scale
    b_vec = np.array([float(x) for x in lines[3].split()[:3]], dtype=float) * scale
    c_vec = np.array([float(x) for x in lines[4].split()[:3]], dtype=float) * scale

    return a_vec, b_vec, c_vec

def main():
    results = []

    for job in job_items:
        conv_path = os.path.join(main_path, job, in_file)
        if not os.path.isfile(conv_path):
            print(f"[WARN] 未找到文件: {conv_path}，跳过该算例。")
            continue

        try:
            a_vec, b_vec, c_vec = read_lattice_from_conv_vasp(conv_path)
        except Exception as e:
            print(f"[ERROR] 读取 {conv_path} 出错: {e}")
            continue

        # 晶格常数（a, b, c 的长度）
        a_len = np.linalg.norm(a_vec)
        b_len = np.linalg.norm(b_vec)
        c_len = np.linalg.norm(c_vec)

        # 夹角：
        # 按你给的定义：
        #   alpha = angle(a, b)
        #   beta  = angle(a, c)
        #   gamma = angle(b, c)
        alpha = angle_between(b_vec, c_vec)  # alpha: b-c 夹角
        beta  = angle_between(a_vec, c_vec)  # beta:  a-c 夹角
        gamma = angle_between(a_vec, b_vec)  # gamma: a-b 夹角

        results.append(
            (job, a_len, b_len, c_len, alpha, beta, gamma)
        )

    # 写入结果文件
    with open(out_file, "w") as f:
        # 注释行（首行）
        f.write("# id  a(Angstrom)  b(Angstrom)  c(Angstrom)  alpha(deg)  beta(deg)  gamma(deg)\n")
        for item in results:
            job, a_len, b_len, c_len, alpha, beta, gamma = item
            f.write(
                f"{job:20s}  "
                f"{a_len:12.6f}  {b_len:12.6f}  {c_len:12.6f}  "
                f"{alpha:12.6f}  {beta:12.6f}  {gamma:12.6f}\n"
            )

    print(f"[INFO] 已写入 {len(results)} 条记录到 {out_file}")

if __name__ == "__main__":
    main()
