#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract potential energy (TOTEN, eV) vs time (fs) from VASP AIMD OUTCAR.

Usage:
    python extract_potential-vs-time_from_OUTCAR.py OUTCAR

If OUTCAR is not given, it will use "./OUTCAR" in current directory.
Output:
    result-potential-vs-time.txt  (two columns: time(fs), E_pot_TOTEN(eV))
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
    # 形如：POTIM  =   0.5000
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


def extract_TOTEN_per_ionic_step(lines):
    """
    提取每个离子步的 free energy TOTEN (eV)。

    逻辑：
      - 遍历行，当遇到包含
          "FREE ENERGIE OF THE ION-ELECTRON SYSTEM"
        的行时，设置一个标志 expect_toten = True；
      - 在 expect_toten = True 的状态下，继续往下读行：
          找到同时包含 "free  energy" 和 "TOTEN" 的行，
          用正则提取该行中的数值（eV），记录到列表，
          然后 expect_toten = False。
    """
    energies = []
    expect_toten = False

    # 匹配 TOTEN 数值，例如：
    # "  free  energy   TOTEN  =     -1048.21674154 eV"
    re_toten = re.compile(r'TOTEN\s*=\s*([0-9Ee+\-\.]+)', re.IGNORECASE)

    for line in lines:
        if 'FREE ENERGIE OF THE ION-ELECTRON SYSTEM' in line:
            expect_toten = True
            continue

        if expect_toten:
            # 找紧跟在 FREE ENERGIE 块后的 free energy TOTEN 行
            if 'free' in line and 'TOTEN' in line:
                m = re_toten.search(line)
                if m:
                    e = float(m.group(1))
                    energies.append(e)
                    expect_toten = False  # 找到后关闭期待
            # 也可以略微宽松：如果碰到空行或其它内容还可以继续，
            # 这里简单起见只在第一次匹配到 TOTEN 时记录。

    return energies


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

    # 2) 提取每个离子步对应的 TOTEN 势能
    energies = extract_TOTEN_per_ionic_step(lines)
    n_ionic = len(energies)
    print(f"Found {n_ionic} ionic steps with TOTEN energy.")

    if n_ionic == 0:
        print("ERROR: No TOTEN data found around 'FREE ENERGIE OF THE ION-ELECTRON SYSTEM' blocks.")
        sys.exit(1)

    if nsw is not None and nsw != n_ionic:
        print(f"WARNING: NSW = {nsw}, but found {n_ionic} TOTEN entries.")
        print("         Will use the number of TOTEN entries (n_ionic) for output.")

    n_steps = n_ionic

    # 3) 生成时间序列 time(fs)：POTIM, 2*POTIM, ..., n_steps*POTIM
    times = [(i + 1) * potim for i in range(n_steps)]

    # 4) 写出到 result-potential-vs-time.txt
    output_file = "result-potential-vs-time.txt"
    with open(output_file, 'w') as fout:
        fout.write("# time(fs)    E_pot_TOTEN(eV)\n")
        for t, e in zip(times, energies):
            fout.write(f"{t:15.6f}  {e:20.8f}\n")

    print(f"Done. Wrote time and potential energy to '{output_file}'.")


if __name__ == "__main__":
    main()
