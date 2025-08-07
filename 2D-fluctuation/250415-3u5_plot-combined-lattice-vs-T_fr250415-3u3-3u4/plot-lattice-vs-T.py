import numpy as np
import matplotlib.pyplot as plt

# Define file paths
apparent_path = "../250415-3u3_compute-triplet_plot-apparent-gr-lattice_fr250415-3/lattice-temperature.txt"
projected_path = "../250415-3u4_plot-projected-gr-lattice_fr250415-3/lattice-temperature.txt"

# Load data
apparent_data = np.loadtxt(apparent_path, comments="#")
projected_data = np.loadtxt(projected_path, comments="#")

# Extract columns
temp_ap, mean_ap, std_ap = apparent_data[:, 0], apparent_data[:, 1], apparent_data[:, 2]
temp_pr, mean_pr, std_pr = projected_data[:, 0], projected_data[:, 1], projected_data[:, 2]

# Plot errorbars
plt.figure()
plt.errorbar(temp_ap, mean_ap, yerr=std_ap, fmt='o-', capsize=4,
             elinewidth=1.2, markerfacecolor='white', label='Apparent')
plt.errorbar(temp_pr, mean_pr, yerr=std_pr, fmt='s--', capsize=4,
             elinewidth=1.2, markerfacecolor='white', label='Projected')

# Plot configuration
plt.xlabel("Temperature (K)", fontsize=15)
plt.ylabel("Lattice constant (\u00c5)", fontsize=15)
plt.grid(True)
plt.ylim(2.42, 2.54)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='upper left', fontsize=15)
plt.tight_layout()


plt.savefig("fig_lattice-T_projected-vs-apparent", dpi=200)
# plt.close()

print("Combined plot saved as lattice_vs_temperature.png")
