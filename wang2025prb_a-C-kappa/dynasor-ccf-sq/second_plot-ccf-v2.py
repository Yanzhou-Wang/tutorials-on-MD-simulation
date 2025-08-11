from dynasor import compute_dynamic_structure_factors, Trajectory
from dynasor.qpoints import get_spherical_qpoints
from dynasor.post_processing import get_spherically_averaged_sample_binned
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import os
from scipy.optimize import curve_fit

mpl.rcParams['text.usetex'] = False

def linear_func(x, a):
    return a * x

def fit_group_velocity(q_vals, omega_vals):
    popt, _ = curve_fit(linear_func, q_vals, omega_vals)
    slope = popt[0]
    v_kms = (2*np.pi) * slope * 1e12 * 1e-10 / 1e3
    return slope, v_kms

values = [3.5]
for index in values:
    path = f"job_{index}_1"
    file_dest = f"{path}/sample_raw.pkl"
    print(file_dest)
    write_dire = "postReuslt-ccf"
    os.makedirs(write_dire, exist_ok=True)

    with open(file_dest, 'rb') as file:
        sample_raw = pickle.load(file)

    sample_averaged = get_spherically_averaged_sample_binned(sample_raw, num_q_bins=9)

    conversion_factor = 1000 / (2*np.pi)
    sample_averaged.omega *= conversion_factor

    q = sample_averaged.q_norms
    f = sample_averaged.omega
    Cl = sample_averaged.Clqw.T
    Ct = sample_averaged.Ctqw.T

    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=200, constrained_layout=True)
    cax = ax.pcolormesh(q, f, sample_averaged.Sqw_coh.T, cmap='Blues', vmin=0, vmax=0.2, shading='auto')
    ax.text(0.68, 0.82, '$S(|\\mathbf{q}|,\\omega)$', transform=ax.transAxes, bbox={'color': 'white', 'alpha': 0.8, 'pad': 3})
    ax.set_xlabel('$|\\mathbf{q}|\\,(\\mathrm{\\AA}^{-1})$')
    ax.set_ylabel('$\\omega/2\\pi\\,(\\mathrm{THz})$')
    ax.set_ylim([0, 60])
    fig.colorbar(cax, ax=ax)
    fig.savefig(f"{write_dire}/sq_{index}.png", format='png', dpi=200)

    fig, axes = plt.subplots(figsize=(3.4, 3.8), nrows=2, dpi=200, sharex=True, sharey=True, constrained_layout=True)

    ax = axes[0]
    cax = ax.pcolormesh(q, f, Cl, cmap='Reds', vmin=0, vmax=0.005, shading='auto')
    ax.text(0.58, 0.25, '$C_L(|\\mathbf{q}|,\\omega)$', transform=ax.transAxes, bbox={'color': 'white', 'alpha': 0.8, 'pad': 3})

    q_mask_L = (q >= 0.0) & (q <= 0.7)
    f_mask_L = (f >= 0.0) & (f <= 50.0)
    q_sub = q[q_mask_L]
    f_sub = f[f_mask_L]
    W_sub = Cl[f_mask_L][:, q_mask_L]

    omega_means_L, q_for_fit_L = [], []
    for j in range(W_sub.shape[1]):
        wj = W_sub[:, j]
        if np.any(wj > 0):
            omega_means_L.append(np.average(f_sub, weights=np.clip(wj, 0, None)))
            q_for_fit_L.append(q_sub[j])
    omega_means_L = np.array(omega_means_L)
    q_for_fit_L = np.array(q_for_fit_L)

    if len(q_for_fit_L) >= 2:
        slope_L, vg_L = fit_group_velocity(q_for_fit_L, omega_means_L)
        xL = np.linspace(q_sub.min(), q_sub.max(), 100)
        ax.plot(xL, linear_func(xL, slope_L), 'b--', lw=2)
        ax.text(0.42, 0.15, rf'$v_g$ = {vg_L:.2f} km/s', transform=ax.transAxes, fontsize=10, va='top', bbox={'color': 'white', 'alpha': 0.8, 'pad': 3})

    ax = axes[1]
    cax = ax.pcolormesh(q, f, Ct, cmap='Oranges', vmin=0, vmax=0.005, shading='auto')
    fig.colorbar(cax, ax=axes).set_label('$C_{L/T}(q,\\omega)$')
    ax.text(0.58, 0.25, '$C_T(|\\mathbf{q}|,\\omega)$', transform=ax.transAxes, bbox={'color': 'white', 'alpha': 0.8, 'pad': 3})
    ax.set_xlabel('$|\\mathbf{q}|\\,(\\mathrm{\\AA}^{-1})$')
    ax.set_ylabel('$\\omega/2\\pi\\,(\\mathrm{THz})$', y=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 15])

    q_mask_T = (q >= 0.0) & (q <= 0.6)
    f_mask_T = (f >= 0.0) & (f <= 50.0)
    q_sub = q[q_mask_T]
    f_sub = f[f_mask_T]
    W_sub = Ct[f_mask_T][:, q_mask_T]

    omega_means_T, q_for_fit_T = [], []
    for j in range(W_sub.shape[1]):
        wj = W_sub[:, j]
        if np.any(wj > 0):
            omega_means_T.append(np.average(f_sub, weights=np.clip(wj, 0, None)))
            q_for_fit_T.append(q_sub[j])
    omega_means_T = np.array(omega_means_T)
    q_for_fit_T = np.array(q_for_fit_T)

    if len(q_for_fit_T) >= 2:
        slope_T, vg_T = fit_group_velocity(q_for_fit_T, omega_means_T)
        xT = np.linspace(q_sub.min(), q_sub.max(), 100)
        ax.plot(xT, linear_func(xT, slope_T), 'b--', lw=2)
        ax.text(0.42, 0.15, rf'$v_g$ = {vg_T:.2f} km/s', transform=ax.transAxes, fontsize=10, va='top', bbox={'color': 'white', 'alpha': 0.8, 'pad': 3})

    fig.savefig(f"{write_dire}/Cqw_{index}.png", format='png', dpi=200)
