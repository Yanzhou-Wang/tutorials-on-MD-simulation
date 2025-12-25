#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract time (fs) and temperature T (K) vs ionic step from VASP AIMD OUTCAR.

Usage:
    python extract_T_vs_time_from_OUTCAR.py OUTCAR

If OUTCAR is not given, it will use "./OUTCAR" in current directory.
Output:
    result-T-vs-time.txt  (two columns: time(fs), T(K))
"""

import sys
import re

def parse_last_NSW_and_POTIM(lines):
    """
    从 OUTCAR 行列表中提取最后一个 NSW 和 POTIM 的值
    """
    nsw = None
    potim = None

    # 形如：NSW = 1000
    re_nsw = re.compile(r'NSW\s*=\s*(\d+)', re.IGNORECASE)
    # 形如：POTIM  =   1.0000
    re_potim = re.compile(r'POTIM\s*=\s*([0-9Ee+\-\.]+)', re.IGNORECASE)

    for line in lines:
        if 'NSW' in line:
            m = re_nsw.search(line)
            if m:
                nsw = int(m.group(1))
        if 'POTIM' in line:
            m = re_potim.search(line)
            if m:
                potim = float(m.group(1))

    return nsw, potim


def extract_temperatures(lines):
    """
    从每个 ionic step 的
      kin. lattice  EKIN_LAT= ... (temperature  xxx.xx K)
    行中提取温度 T(K)。

    筛选规则：
      - 行中包含 'EKIN_LAT'
      - 同时包含 '(temperature'
      - 用正则匹配括号里的温度值 xxx.xx
    """
    temps = []

    # 匹配括号中的温度：例如 (temperature  300.00 K)
    re_temp = re.compile(
        r'\(temperature\s+([0-9Ee+\-\.]+)\s*K\)',
        re.IGNORECASE
    )

    for line in lines:
        # 更严格一点：确保是 kin. lattice EKIN_LAT 行
        if 'EKIN_LAT' in line and '(temperature' in line:
            m = re_temp.search(line)
            if m:
                T = float(m.group(1))
                temps.append(T)

    return temps


def main():
    if len(sys.argv) > 1:
        outcar_path = sys.argv[1]
    else:
        outcar_path = "OUTCAR"

    # 读取 OUTCAR
    try:
        with open(outcar_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"ERROR: Cannot find file '{outcar_path}'")
        sys.exit(1)

    # 1) 提取最后一个 NSW 和 POTIM
    nsw, potim = parse_last_NSW_and_POTIM(lines)
    if nsw is None or potim is None:
        print("ERROR: Failed to parse NSW or POTIM from OUTCAR.")
        print(f"  Parsed NSW = {nsw}, POTIM = {potim}")
        sys.exit(1)

    print(f"Parsed NSW   = {nsw}")
    print(f"Parsed POTIM = {potim} fs")

    # 2) 提取每个 ionic step 的温度（来自 EKIN_LAT 行）
    temps = extract_temperatures(lines)
    n_ionic = len(temps)
    print(f"Found {n_ionic} ionic steps with temperature information.")

    if n_ionic == 0:
        print("ERROR: No temperature data found in OUTCAR (no EKIN_LAT temperature lines).")
        sys.exit(1)

    if nsw is not None and nsw != n_ionic:
        print(f"WARNING: NSW = {nsw}, but found {n_ionic} temperature entries (EKIN_LAT lines).")
        print("         Will use the number of temperature entries (n_ionic) for output.")

    # 使用温度条目数作为步数
    n_steps = n_ionic

    # 时间从 POTIM, 2*POTIM, ..., n_steps*POTIM
    times = [(i + 1) * potim for i in range(n_steps)]

    # 3) 写出到 result-T-vs-time.txt
    output_file = "result-T-vs-time.txt"
    with open(output_file, 'w') as fout:
        fout.write("# time(fs)    T(K)\n")
        for t, T in zip(times, temps):
            fout.write(f"{t:15.6f}  {T:15.6f}\n")

    print(f"Done. Wrote time and temperature to '{output_file}'.")


if __name__ == "__main__":
    main()
