import numpy as np
import matplotlib.pyplot as plt
#from ase import Atoms
#from ase.build import bulk
#from dynasor import compute_dynamic_structure_factors, Trajectory
#from dynasor.qpoints import get_supercell_qpoints_along_path
#from seekpath import get_path
from dynasor import read_sample_from_npz
from dynasor.units import radians_per_fs_to_meV as conversion_factor





sample = read_sample_from_npz('dat_sample.npz')
q_segments = np.load('dat_q_segments.npy', allow_pickle=True)
path = np.load('dat_path.npy', allow_pickle=True)



q_distances = []
q_labels = dict()

# starting point
qr = 0.0

# collect labels and q-distances along the entire q-point path
for it, q_segment in enumerate(q_segments):
    q_distances.append(qr)
    q_labels[qr] = path[it][0]
    for qi, qj in zip(q_segment[1:], q_segment[:-1]):
        qr += np.linalg.norm(qi - qj)
        q_distances.append(qr)

q_labels[qr] = path[-1][1]
q_distances = np.array(q_distances)



fig = plt.figure(figsize=(5.2, 2.8), dpi=140)
ax = fig.add_subplot(111)
ax.pcolormesh(q_distances, conversion_factor * sample.omega,
              sample.Sqw_coh.T, cmap='Blues', vmin=0, vmax=4)

xticks = []
xticklabels = []
for q_dist, q_label in q_labels.items():
    ax.axvline(q_dist, c='0.5', alpha=0.5, ls='--')
    xticks.append(q_dist)
    xticklabels.append(q_label.replace('GAMMA', r'$\Gamma$'))

ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlim([0, q_distances.max()])
ax.set_ylim([0, 55])
ax.set_ylabel('Frequency (meV)')
fig.tight_layout()
plt.savefig('fig_s_qw.png', dpi=200)





fig = plt.figure(figsize=(5.2, 2.8), dpi=140)
ax = fig.add_subplot(111)
ax.pcolormesh(q_distances, conversion_factor * sample.omega,
              sample.Clqw.T - sample.Ctqw.T,
              cmap='RdBu', vmin=-6000, vmax=6000)

ax.plot([0, 1.0], [0, 36], alpha=0.5, ls='--', c='0.3', lw=2)
ax.plot([0, 1.5], [0, 32], alpha=0.5, ls='--', c='0.3', lw=2)

ax.text(0.02, 0.89, r'$C_{T-L}(\mathbf{q}, \omega)$', transform=ax.transAxes,
        bbox={'color': 'white', 'alpha': 0.8, 'pad': 3})

xticks = []
xticklabels = []
for q_dist, q_label in q_labels.items():
    ax.axvline(q_dist, c='0.5', alpha=0.5, ls='--')
    xticks.append(q_dist)
    xticklabels.append(q_label.replace('GAMMA', r'$\Gamma$'))

ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlim([0, q_distances.max()])
ax.set_ylim([0, 55])
ax.set_ylabel('Frequency (meV)')
fig.tight_layout()
plt.savefig('fig_clt_qw.png', dpi=200)
