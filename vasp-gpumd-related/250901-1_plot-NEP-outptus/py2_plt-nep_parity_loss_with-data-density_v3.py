#!/usr/bin/env python3


"""
Usage: ./py_xxx.py
PS:  The script will automatically read data from a called "job_1" directory  and plot relevant figures
PS: "job_1" and "py_xxx.py" must stay in same architecture
"""


# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Global plotting parameters
lw = 1.8
fs = 15
ps = 50

color_map = {
    'energy': 'tab:blue',
    'force': 'tab:orange',
    'virial': 'tab:green',
}

# colormap for density
density_cmap = 'coolwarm'

write_dire = 'result-plotted-nep_with-data-density'
os.makedirs(write_dire, exist_ok=True)

r_main_path = './'
job_index = [1]


# %%
def get_point_density(x, y, bins=200):
    """
    Estimate point density using a 2D histogram.
    Return one density value for each (x, y) point.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # avoid zero-width range
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    if x_min == x_max:
        x_min -= 1e-8
        x_max += 1e-8
    if y_min == y_max:
        y_min -= 1e-8
        y_max += 1e-8

    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[x_min, x_max], [y_min, y_max]])

    # find which bin each point belongs to
    xbin = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    ybin = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)

    density = hist[xbin, ybin]

    # avoid zero values for LogNorm
    density[density < 1] = 1
    return density


# %%
def plot_loss(read_dire, index):
    loss_path = os.path.join(read_dire, 'loss.out')
    if os.path.isfile(loss_path):
        loss = np.loadtxt(loss_path)
        generation = np.arange(1, len(loss) + 1) * 100
        plt.figure()
        plt.loglog(generation, loss[:, 1:4], '-.', linewidth=lw)
        plt.loglog(generation, loss[:, 4:7], ':', linewidth=lw * 3)
        if np.array_equal(loss[:, 7], loss[:, 8]) and np.array_equal(loss[:, 8], loss[:, 9]):
            plt.legend(['Total', 'L1', 'L2', 'RMSE$^{e-train}$', 'RMSE$^{f-train}$', 'RMSE$^{v-train}$'],
                       loc='upper left', bbox_to_anchor=(1, 1), fontsize=fs-3)
        else:
            plt.loglog(generation, loss[:, 7])
            plt.loglog(generation, loss[:, 8])
            plt.loglog(generation, loss[:, 9])
            plt.legend(['Total', 'L1', 'L2', 'RMSE$^{e-train}$', 'RMSE$^{f-train}$', 'RMSE$^{v-train}$',
                        'RMSE$^{e-test}$', 'RMSE$^{f-test}$', 'RMSE$^{v-test}$'],
                       loc='upper left', bbox_to_anchor=(1, 1), fontsize=fs-3)

        plt.xlabel('Generation', fontsize=fs)
        plt.ylabel('Loss', fontsize=fs)
        plt.xticks(fontsize=fs-3)
        plt.yticks(fontsize=fs-3)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(write_dire, f'loss_{index}.png'), dpi=200, bbox_inches='tight')
        # plt.show()
        #plt.close()
    else:
        print(f'"loss.out" is missing in the directory {read_dire}')


for i in job_index:
    index = i
    read_dire = os.path.join(r_main_path, f'job_{i}')
    plot_loss(read_dire, i)


# %%
def plot_energy(read_dire, index, file_tag):
    filename = f'energy_{file_tag}.out'
    path = os.path.join(read_dire, filename)
    if os.path.isfile(path):
        energy = np.loadtxt(path)
        dft = energy[:, 1]
        nep = energy[:, 0]

        density = get_point_density(dft, nep, bins=200)
        order = np.argsort(density)
        dft = dft[order]
        nep = nep[order]
        density = density[order]

        plt.figure()
        sc = plt.scatter(dft, nep, c=density, s=ps, cmap=density_cmap, norm=LogNorm(), edgecolors='none')

        range_broad = np.ptp(dft)
        upper = dft.max() + range_broad / 8
        lower = dft.min() - range_broad / 8
        plt.plot([lower, upper], [lower, upper], '--', color='k', linewidth=lw-1)

        rmse = np.sqrt(np.mean(np.square(dft - nep)))
        rmse_str = f'RMSE_e = {round(rmse, 3)}'
        mae = np.mean(np.abs(dft - nep))
        mae_str = f'MAE_e = {round(mae, 3)}'

        tx = lower + (upper - lower) / 15
        ty = lower + (upper - lower) * 13 / 15
        plt.text(tx, ty, f'{rmse_str}\n{mae_str}', fontsize=fs)

        plt.xlabel('DFT energy (eV/atom)', fontsize=fs)
        plt.ylabel('NEP energy (eV/atom)', fontsize=fs)
        plt.xticks(fontsize=fs-3)
        plt.yticks(fontsize=fs-3)
        plt.axis('square')
        plt.grid(True)

        cbar = plt.colorbar(sc)
        cbar.set_label('Data density', fontsize=fs)
        cbar.ax.tick_params(labelsize=fs-3)

        plt.tight_layout()
        plt.savefig(os.path.join(write_dire, f'{file_tag}-energy_{index}.png'), dpi=200, bbox_inches='tight')
        # plt.show()
        #plt.close()
    else:
        print(f'{filename} is missing in the directory {read_dire}')


for i in job_index:
    index = i
    read_dire = os.path.join(r_main_path, f'job_{i}')
    plot_energy(read_dire, index, 'train')
    plot_energy(read_dire, index, 'test')


# %%
def plot_force(read_dire, index, file_tag):
    filename = f'force_{file_tag}.out'
    path = os.path.join(read_dire, filename)
    if os.path.isfile(path):
        force = np.loadtxt(path)
        dft = force[:, 3:6].reshape(-1)
        nep = force[:, 0:3].reshape(-1)

        density = get_point_density(dft, nep, bins=200)
        order = np.argsort(density)
        dft = dft[order]
        nep = nep[order]
        density = density[order]

        plt.figure()
        sc = plt.scatter(dft, nep, c=density, s=ps, cmap=density_cmap, norm=LogNorm(), edgecolors='none')

        range_broad = np.ptp(dft)
        upper = dft.max() + range_broad / 8
        lower = dft.min() - range_broad / 8
        plt.plot([lower, upper], [lower, upper], '--', color='k', linewidth=lw-1)

        rmse = np.sqrt(np.mean(np.square(dft - nep)))
        rmse_str = f'RMSE_f = {round(rmse, 3)}'
        mae = np.mean(np.abs(dft - nep))
        mae_str = f'MAE_f = {round(mae, 3)}'

        tx = lower + (upper - lower) / 15
        ty = lower + (upper - lower) * 13 / 15
        plt.text(tx, ty, f'{rmse_str}\n{mae_str}', fontsize=fs)

        plt.xlabel('DFT force (eV/\u212B)', fontsize=fs)
        plt.ylabel('NEP force (eV/\u212B)', fontsize=fs)
        plt.xticks(fontsize=fs-3)
        plt.yticks(fontsize=fs-3)
        plt.axis('square')
        plt.grid(True)

        cbar = plt.colorbar(sc)
        cbar.set_label('Data density', fontsize=fs)
        cbar.ax.tick_params(labelsize=fs-3)

        plt.tight_layout()
        plt.savefig(os.path.join(write_dire, f'{file_tag}-force_{index}.png'), dpi=200, bbox_inches='tight')
        # plt.show()
        #plt.close()
    else:
        print(f'{filename} is missing in the directory {read_dire}')


for i in job_index:
    index = i
    read_dire = os.path.join(r_main_path, f'job_{i}')
    plot_force(read_dire, index, 'train')
    plot_force(read_dire, index, 'test')


# %%
def plot_virial(read_dire, index, file_tag):
    filename = f'virial_{file_tag}.out'
    path = os.path.join(read_dire, filename)
    if os.path.isfile(path):
        virial = np.loadtxt(path)
        nep = virial[:, 0:6].reshape(-1)
        dft = virial[:, 6:12].reshape(-1)

        mask = dft != -1000000
        dft = dft[mask]
        nep = nep[mask]

        density = get_point_density(dft, nep, bins=200)
        order = np.argsort(density)
        dft = dft[order]
        nep = nep[order]
        density = density[order]

        plt.figure()
        sc = plt.scatter(dft, nep, c=density, s=ps, cmap=density_cmap, norm=LogNorm(), edgecolors='none')

        range_broad = np.ptp(dft)
        upper = dft.max() + range_broad / 8
        lower = dft.min() - range_broad / 8
        plt.plot([lower, upper], [lower, upper], '--', color='k', linewidth=lw-1)

        rmse = np.sqrt(np.mean(np.square(dft - nep)))
        rmse_str = f'RMSE_v = {round(rmse, 3)}'
        mae = np.mean(np.abs(dft - nep))
        mae_str = f'MAE_v = {round(mae, 3)}'

        tx = lower + (upper - lower) / 15
        ty = lower + (upper - lower) * 13 / 15
        plt.text(tx, ty, f'{rmse_str}\n{mae_str}', fontsize=fs)

        plt.xlabel('DFT virial (eV/atom)', fontsize=fs)
        plt.ylabel('NEP virial (eV/atom)', fontsize=fs)
        plt.xticks(fontsize=fs-3)
        plt.yticks(fontsize=fs-3)
        plt.axis('square')
        plt.grid(True)

        cbar = plt.colorbar(sc)
        cbar.set_label('Data density', fontsize=fs)
        cbar.ax.tick_params(labelsize=fs-3)

        plt.tight_layout()
        plt.savefig(os.path.join(write_dire, f'{file_tag}-virial_{index}.png'), dpi=200, bbox_inches='tight')
        # plt.show()
        #plt.close()
    else:
        print(f'{filename} is missing in the directory {read_dire}')


for i in job_index:
    index = i
    read_dire = os.path.join(r_main_path, f'job_{i}')
    plot_virial(read_dire, index, 'train')
    plot_virial(read_dire, index, 'test')
# %%
