# %%
import os
import numpy as np
import matplotlib.pyplot as plt

# Global plotting parameters
lw = 1.8
fs = 15
ps = 15
write_dire = 'result-python-plotting'
os.makedirs(write_dire, exist_ok=True)

r_main_path = './'
job_index = [1]



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
            plt.legend(['Total', 'L1', 'L2', 'RMSE$^{e-train}$', 'RMSE$^{f-train}$', 'RMSE$^{v-train}$'], loc='upper left', bbox_to_anchor=(1, 1), fontsize=fs-3)
        else:
            plt.loglog(generation, loss[:, 7])
            plt.loglog(generation, loss[:, 8])
            plt.loglog(generation, loss[:, 9])
            plt.legend(['Total', 'L1', 'L2', 'RMSE$^{e-train}$', 'RMSE$^{f-train}$', 'RMSE$^{v-train}$',
                        'RMSE$^{e-test}$', 'RMSE$^{f-test}$', 'RMSE$^{v-test}$'], loc='upper left', bbox_to_anchor=(1, 1), fontsize=fs-3)    
        
        plt.xlabel('Generation', fontsize=fs)
        plt.ylabel('Loss', fontsize=fs)
        plt.xticks(fontsize=fs-3)
        plt.yticks(fontsize=fs-3)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(write_dire, f'loss_{index}.png'), dpi=200, bbox_inches='tight')
        plt.show()
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
        plt.figure()
        plt.plot(energy[:, 1], energy[:, 0], '.', markersize=ps)
        range_broad = np.ptp(energy[:, 1])
        upper = energy[:, 1].max() + range_broad / 8
        lower = energy[:, 1].min() - range_broad / 8
        plt.plot([lower, upper], [lower, upper], '--', color='k', linewidth=lw)
        rmse = np.sqrt(np.mean(np.square(energy[:, 1] - energy[:, 0])))
        rmse_str = f'RMSE_e = {round(rmse, 3)}'
        mae = np.mean(np.abs(energy[:, 1] - energy[:, 0]))
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
        plt.tight_layout()
        plt.savefig(os.path.join(write_dire, f'{file_tag}-energy_{index}.png'), dpi=200, bbox_inches='tight')
        plt.show()
    else:
        print(f'{filename} is missing in the directory {read_dire}')


# Main loop
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
        plt.figure()
        plt.plot(dft, nep, '.', markersize=ps)
        range_broad = np.ptp(dft)
        upper = dft.max() + range_broad / 8
        lower = dft.min() - range_broad / 8
        plt.plot([lower, upper], [lower, upper], '--', color='k', linewidth=lw)
        rmse=np.sqrt(np.mean(np.square(dft - nep)))
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
        plt.tight_layout()
        plt.savefig(os.path.join(write_dire, f'{file_tag}-force_{index}.png'), dpi=200, bbox_inches='tight')
        plt.show()
    else:
        print(f'{filename} is missing in the directory {read_dire}')
        

# Main loop
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
        plt.figure()
        plt.plot(dft, nep, '.', markersize=ps)
        range_broad = np.ptp(dft)
        upper = dft.max() + range_broad / 8
        lower = dft.min() - range_broad / 8
        plt.plot([lower, upper], [lower, upper], '--', color='k', linewidth=lw)
        rmse=np.sqrt(np.mean(np.square(dft-nep)))
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
        plt.tight_layout()
        plt.savefig(os.path.join(write_dire, f'{file_tag}-virial_{index}.png'), dpi=200, bbox_inches='tight')
        plt.show()
    else:
        print(f'{filename} is missing in the directory {read_dire}')
        


# Main loop
for i in job_index:
    index = i
    read_dire = os.path.join(r_main_path, f'job_{i}')
    plot_virial(read_dire, index, 'train')
    plot_virial(read_dire, index, 'test')
# %%