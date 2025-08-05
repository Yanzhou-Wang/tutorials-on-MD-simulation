import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

# Settings
is_list = [100, 400, 700, 1000, 1300, 1600, 1900, 2300, 2700, 3100, 3500]
js_list = [25200]
ks_list = [1, 2, 3, 4, 5, 6]

job_root = "../250415-3_graphene-lattice-vs-T_npt-scr_25200atom_denser-traj_6cycles_fr241013-1u"
output_dir = "result-figures_projected-lattice-vs-T"
os.makedirs(output_dir, exist_ok=True)

mean_lattice = []
std_lattice = []

for i in is_list:
    lattice_list = []
    for j in js_list:
        for k in ks_list:
            job_dir = f"job_{i}_{j}_{k}"
            dump_path = os.path.join(job_root, job_dir, "dump.xyz")
            if not os.path.exists(dump_path):
                print(f"Missing: {dump_path}")
                continue
            try:
#                frames = read(dump_path, index=slice(0, 50))  # read first 50 frames
                frames = read(dump_path, index=slice(0, 500, 10))  # read first 50 frames
#                frames = read(dump_path, index="0:1:50", format="extxyz")
                for atoms in frames:
                    lx = atoms.get_cell()[0, 0]  # assume orthogonal box
                    projected = lx / 105.0          # 105 is replicates along x-axis direction
                    lattice_list.append(projected)
            except Exception as e:
                print(f"Failed to read {dump_path}: {e}")
                continue

    lattice_array = np.array(lattice_list)
    print(f"Temperature {i} K: {len(lattice_array)} projected lattice constants")

    if lattice_array.size != 300:       #50x6
        print(f"Warning: Expected 300 values (50 frames * 6 runs), got {lattice_array.size}")

    try:
        #matrix = lattice_array.reshape((50, 6))  # shape: (50 rows, 6 cols)
        #col_means = matrix.mean(axis=0)  # get 6 mean values
        mean_lattice.append(np.mean(lattice_array))
        std_lattice.append(np.std(lattice_array))
    except Exception as e:
        print(f"Reshape failed at T={i}K with {lattice_array.size} entries: {e}")
        mean_lattice.append(np.nan)
        std_lattice.append(np.nan)

# Save data to file in current directory
output_data_path = "lattice-temperature.txt"
with open(output_data_path, 'w') as f:
    f.write("# Temperature(K)  Mean_Lattice(Å)  Std_Deviation(Å)\n")
    for temp, mean_val, std_val in zip(is_list, mean_lattice, std_lattice):
        f.write(f"{temp:6d}  {mean_val:.6f}  {std_val:.6f}\n")

# Plot errorbar
plt.figure()
plt.errorbar(is_list, mean_lattice, yerr=std_lattice, fmt='o-', capsize=4,
             elinewidth=1.2, markerfacecolor='white')
plt.xlabel("Temperature (K)", fontsize=15)
plt.ylabel("Lattice constant (Å)", fontsize=15)
plt.grid(True)
plt.ylim(2.40, 2.55)
#plt.ylim(2.375, 2.575)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "lattice_vs_temperature.png"), dpi=200)
# plt.close()

print(f"Saved plot to {output_dir}")
print(f"Saved data to {output_data_path}")
