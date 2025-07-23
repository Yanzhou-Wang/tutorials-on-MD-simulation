import numpy as np
import matplotlib.pyplot as plt

# 读取数据
filepath = "result-elastic/module-vs-temperature.txt"
data = []
with open(filepath, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.split()
        data.append([float(x) for x in parts])

# 转换为 numpy 数组
data = np.array(data)

# 提取各列
T = data[:, 0]
Kv = data[:, 1]
Gv = data[:, 2]
E = data[:, 3]
Kv_ste = data[:, 5]
Gv_ste = data[:, 6]
E_ste = data[:, 7]

# 绘图
fs=15

plt.figure(figsize=(8,6))

plt.errorbar(T, Kv, yerr=Kv_ste, fmt='o-', color='tab:blue', label='Bulk Modulus ' + r"$B$")
plt.errorbar(T, Gv, yerr=Gv_ste, fmt='s-', color='tab:orange', label='Shear Modulus ' + r"$G$")
plt.errorbar(T, E, yerr=E_ste, fmt='^-', color='tab:green', label="Young's Modulus " + r"$E$")

plt.xlabel("Temperature T (K)",fontsize=fs)
plt.ylabel("Modulus (GPa)",fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.ylim([0,350])

plt.legend(fontsize=fs)
plt.title("333-FeCo", fontsize=fs)
plt.tight_layout()

plt.savefig("result-elastic/module-vs-T.png", dpi=200)
plt.show()

print("✅ Plot saved as result-elastic/module-vs-T.png")
