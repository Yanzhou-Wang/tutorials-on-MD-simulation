import os
import numpy as np
import matplotlib.pyplot as plt
#from ase.io import read
from calorine.gpumd import read_thermo

# Settings
is_list = [100, 400, 700, 1000, 1300, 1600, 1900, 2300, 2700, 3100, 3500]
js_list = [25200]
ks_list = [1, 2, 3, 4, 5, 6]

replicates_along_zigzag = 105  # for 25200 atoms

n_frames_last = 500

#n_frames_dumpxyz = 500
#sample_interval = 10
#sample_start_index = 0

selected_temps = [400, 1000, 1600, 2300, 3500]  # for histgram

color_map = {
    400: "tab:blue",
    1000: "tab:orange",
    1600: "tab:green",
    2300: "tab:red",
    3500: "tab:purple"
}


job_root = "../250415-3_graphene-lattice-vs-T_npt-scr_25200atom_denser-traj_6cycles_fr241013-1u"
output_dir = "./"
os.makedirs(output_dir, exist_ok=True)

mean_lattice = []
std_lattice = []

for i in is_list:
    lattice_list = []
    for j in js_list:
        for k in ks_list:
            job_dir = f"job_{i}_{j}_{k}"
            thermo_path = os.path.join(job_root, job_dir, "thermo.out")
            if not os.path.exists(thermo_path):
                print(f"Missing: {thermo_path}")
                continue
            try:
                thermos = read_thermo(thermo_path)
                if len(thermos) < n_frames_last:
                    print(f"Warning: only {len(thermos)} frames in {thermo_path}")
                    continue
                thermos_selected = thermos.iloc[-n_frames_last:]
                lxs = thermos_selected['cell_xx'].values
                projected_l = lxs / replicates_along_zigzag
                lattice_list.extend(projected_l)
                lattice_array = np.array(lattice_list)
            except Exception as e:
                print(f"Error reading {thermo_path}: {e}")
                continue

    if i in selected_temps:
        #plt.hist(lattice_array, bins=100, density=True, alpha=0.6, edgecolor='black', label=f"{i} K", color=color_map[i])
        plt.hist(lattice_array, bins=20, density=True, alpha=0.6,
         edgecolor='black', label=f"{i} K", color=color_map[i])



# Finalize and save combined histogram figure
plt.xlabel("Projected lattice constant (Å)", fontsize=15)
plt.ylabel("Probability", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(True)
plt.legend()
plt.xlim(2.44, 2.467)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_histogram-projected-lattice_vary-T.png"), dpi=200)
# plt.close()



'''
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
plt.savefig(os.path.join(output_dir, "fig_projected-lattice-T.png"), dpi=200)
# plt.close()

print(f"Saved plot to {output_dir}")
print(f"Saved data to {output_data_path}")
'''