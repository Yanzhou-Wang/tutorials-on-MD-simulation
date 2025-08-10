import os
import numpy as np
import matplotlib.pyplot as plt
import sys

# Set parameters
is_list = [100, 400, 700, 1000, 1300, 1600, 1900, 2300, 2700, 3100, 3500]
js_list = [25200]  # Atom counts (reserved for future extensions)
ks_list = [1, 2, 3, 4, 5, 6]
n_frames = 50
n_fold_per_atom = 3   # coordination number for center atom



# Output directory
output_dir = "./"
os.makedirs(output_dir, exist_ok=True)

# Temperatures selected for histogram plotting
selected_temps = [400, 1000, 1600, 2300, 3500]
color_map = {
    400: "tab:blue",
    1000: "tab:orange",
    1600: "tab:green",
    2300: "tab:red",
    3500: "tab:purple"
}

# Store mean and std for each temperature
mean_lattice = []
std_lattice = []

# Initialize combined histogram figure
plt.figure()

# Loop over temperatures
for i in is_list:
    lattice_all = []
    valid_count = 0

    for j in js_list:
        for k in ks_list:
            job_dir = f"job_{i}_{j}_{k}"
            bond_path = os.path.join(job_dir, "bond.txt")
            if not os.path.exists(bond_path):
                print(f"Missing file: {bond_path}")
                continue

            try:
                raw_data = np.loadtxt(bond_path, usecols=(0, 1, 3))
                filtered = raw_data[(raw_data[:,2] > raw_data[:,0]) & (raw_data[:,2] > raw_data[:,1])]
                lattice_all.extend(filtered[:, 2])
                valid_count += filtered.shape[0]
            except Exception as e:
                print(f"Error reading {bond_path}: {e}")
                continue

    lattice_all = np.array(lattice_all)
    print(f"Temperature {i} K: valid lattice constants = {valid_count}")

    max_count = j * n_fold_per_atom * len(ks_list) * n_frames  # 22680000
    if lattice_all.size > max_count:
        lattice_all = lattice_all[:max_count]

    if lattice_all.size == 0:
        mean_lattice.append(np.nan)
        std_lattice.append(np.nan)
        continue

    if i in selected_temps:
        plt.hist(lattice_all, bins=100, density=True, alpha=0.6,
                 edgecolor='black', label=f"{i} K", color=color_map[i])

    # compute error using matrix row-averaging method
    total = lattice_all.size
    n = j * n_fold_per_atom
    m = n_frames * len(ks_list)   # m=50 frames * 6 runs, n=25200 atoms * 3 lattice/atom)  # take single frame as a basic unit for statistical deviation
    try:
        """
        vector = lattice_all.reshape(total, -1)
        mean_lattice.append(np.mean(vector))
        std_lattice.append(np.std(vector))
        """
        
        """
        matrix = lattice_all.reshape((n, m))
        mean_lattice.append(np.mean(lattice_all))
        std_one_frame_lattice = np.std(matrix, axis=0)
        std_mean_lattice = np.mean(std_one_frame_lattice)
        std_lattice.append(std_mean_lattice)
        """
        
        
        matrix = lattice_all.reshape((n, m))
        mean_lattice.append(np.mean(lattice_all))
        std_mean = np.mean(matrix, axis=1)
        std_mean_lattice = np.std(std_mean)
        std_lattice.append(std_mean_lattice)
        
        
        
    except ValueError as e:
        print(f"Error: Cannot reshape lattice_all of size {lattice_all.size} to shape ({n}, {m})")
        print(f"Exception: {e}")
        sys.exit()

    
    
# Finalize and save combined histogram figure
plt.axvline(x=2.47, color='black', linestyle='--', linewidth=1.2, label=r'$a_{400} = 2.47$')
plt.axvline(x=2.5, color='black', linestyle='-.', linewidth=1.2, label=r'$a_{3500} = 2.5$')
plt.xlabel("Apparent lattice constant (Å)", fontsize=15)
plt.ylabel("Probability", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(True)
plt.legend()
plt.xlim(2.1, 2.9)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_hist-apparent-lattice_vary-T.png"), dpi=200)
# plt.close()







# Plot errorbar
plt.figure()
plt.errorbar(is_list, mean_lattice, yerr=std_lattice, fmt='o-', capsize=4,
             elinewidth=1.2, markerfacecolor='white')
plt.xlabel("Temperature (K)", fontsize=15)
plt.ylabel("Lattice constant (Å)", fontsize=15)
plt.grid(True)
plt.ylim(2.4, 2.54)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig(os.path.join(output_dir, "fig_lattice_vs_temperature.png"), dpi=200)
# plt.close()

# Save lattice-temperature data to current directory
with open("lattice-temperature.txt", 'w') as f:
    f.write("# Temperature(K)  Mean_Lattice(Å)  Std_Deviation(Å)\n")
    for temp, mean_val, std_val in zip(is_list, mean_lattice, std_lattice):
        f.write(f"{temp:6d}  {mean_val:.6f}  {std_val:.6f}\n")

print(f"All plots saved in: {output_dir}")
print("Saved data to ./lattice-temperature.txt")
