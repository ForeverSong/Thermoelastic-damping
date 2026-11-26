"""Kinetic figures: (1) damping curves and error maps of the closures in the (Omega, Kn) plane,
(2) phase-space structure of the kinetic wall layer.  Writes ../figures/fig_bte.*, fig_bte_phase.*"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import bte1d as b, gkted as g

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, '..', 'figures')
C1, C2, C3, C4 = '#257de0', '#f47230', '#00b894', '#f2a600'
GREY, INK, INK2 = '#8c8c8c', '#191919', '#666666'
plt.rcParams.update({'font.size': 8.5, 'axes.labelsize': 8.5, 'legend.fontsize': 7.5, 'axes.edgecolor': INK2,
                     'axes.linewidth': 0.6, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK,
                     'text.color': INK, 'lines.linewidth': 1.5, 'savefig.dpi': 300})
SEQ = LinearSegmentedColormap.from_list('seq', ['#f4f8fd', '#b9d6f5', '#5a9de8', '#1f5fbf', '#0b2e6b'])
DIV = LinearSegmentedColormap.from_list('div', ['#0b3d91', '#257de0', '#a9c9f0', '#f2f2f0', '#f8b995', '#f47230', '#8a2f00'])
R = json.load(open(os.path.join(HERE, '..', 'bte_results.json')))

# ---------------------------------------------------------------- error maps
Om = np.logspace(-1, 2.5, 71); Kn = np.logspace(-2, 0, 61)
ES = np.zeros((len(Kn), len(Om))); EC = np.zeros_like(ES)
for i, k in enumerate(Kn):
    for j, o in enumerate(Om):
        ref = b.bte_specular(o, k, N=20000); c = b.closures(o, k)
        ES[i, j] = abs(c['scalar'] - ref) / abs(1 + ref); EC[i, j] = abs(c['consistent'] - ref) / abs(1 + ref)
R['map'] = dict(frac_scalar_below_1pct=float(np.mean(ES < 1e-2)), frac_consistent_below_1pct=float(np.mean(EC < 1e-2)),
                Kn_1pct_scalar_at_Om3=float(Kn[np.argmax(ES[:, np.argmin(abs(Om - 3))] > 1e-2)]),
                Kn_1pct_consistent_at_Om3=float(Kn[np.argmax(EC[:, np.argmin(abs(Om - 3))] > 1e-2)]))
print(R['map']); json.dump(R, open(os.path.join(HERE, '..', 'bte_results.json'), 'w'), indent=1)
fig = plt.figure(figsize=(6.9, 2.55)); gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.32, left=0.075, right=0.9, top=0.88, bottom=0.19)
ax = fig.add_subplot(gs[0]); rows = R['curves']['0.3']; O = [r['Om'] for r in rows]
ax.loglog(O, [r['fourier'] for r in rows], color=GREY, lw=1.0, ls=(0, (4, 2)), label='Fourier')
ax.loglog(O, [r['scalar'] for r in rows], color=C1, label=r'scalar G-K')
ax.loglog(O, [r['consistent'] for r in rows], color=C2, ls='--', label=r'consistent G-K')
ax.loglog(O[::2], [r['bte_spec'] for r in rows][::2], ls='none', marker='o', ms=3.4, color=INK, label='BTE, specular')
ax.loglog(O[1::2], [r['bte_diff'] for r in rows][1::2], ls='none', marker='s', ms=3.4, mfc='white', color=INK, mew=0.9, label='BTE, diffuse')
ax.set_ylim(1.5e-4, 1.5); ax.set_xlabel(r'$\Omega=\omega h^2/\chi$'); ax.set_ylabel(r'$Q^{-1}/\Delta_E$'); ax.set_title('(a) Kn $=0.3$', fontsize=8.5, loc='left')
ax.legend(frameon=False, fontsize=6.4, loc='lower left', handlelength=1.9); ax.grid(True, which='major', color='#e6e5e0', lw=0.5); ax.set_axisbelow(True)
norm = LogNorm(1e-6, 1.0)
for k, (E, tag) in enumerate(((ES, '(b) error of the scalar closure'), (EC, '(c) error of the consistent closure'))):
    ax = fig.add_subplot(gs[k + 1]); im = ax.pcolormesh(Om, Kn, np.clip(E, 1e-6, 1), cmap=SEQ, norm=norm, shading='auto', rasterized=True)
    cs = ax.contour(Om, Kn, E, levels=[1e-4, 1e-2], colors=[INK], linewidths=[0.6, 1.0], linestyles=['--', '-'])
    ax.clabel(cs, fmt={1e-4: r'$10^{-4}$', 1e-2: '1%'}, fontsize=6.5)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$\Omega$'); ax.set_title(tag, fontsize=8.5, loc='left')
    if k == 0: ax.set_ylabel('Kn')
cax = fig.add_axes([0.915, 0.19, 0.012, 0.69]); cb = fig.colorbar(im, cax=cax); cb.set_label(r'$|I-I_{\rm BTE}|/|1+I_{\rm BTE}|$', fontsize=7.5); cb.ax.tick_params(labelsize=7)
fig.savefig(os.path.join(FIG, 'fig_bte.pdf')); fig.savefig(os.path.join(FIG, 'fig_bte.png'), dpi=160); plt.close(fig)

# ---------------------------------------------------------------- phase space near the wall
Omv, Knv = 10.0, 0.1; Lam, L, tt = b.gk_params(Omv, Knv); m = np.sqrt(Lam * (1 + tt) / (1 + 3 * Lam * L ** 2))
fig = plt.figure(figsize=(6.9, 2.6)); gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.045, 0.3, 1.0], wspace=0.22, left=0.085, right=0.985, top=0.88, bottom=0.18)
prof = {}
for k, wall in enumerate(('specular', 'diffuse')):
    F = b.bte_dom(Omv, Knv, wall, N=1200, Nmu=32, return_fields=True)
    Gc = 0.5 * (F['G'][:, 1:] + F['G'][:, :-1]); o = np.argsort(F['mu']); A = np.imag(Gc[o] - F['T'][None, :])
    ax = fig.add_subplot(gs[k]); d = 0.5 - F['zc']; sel = d < 0.3
    im = ax.pcolormesh(d[sel] / Knv, F['mu'][o], A[:, sel] / np.abs(A[:, sel]).max(), cmap=DIV, vmin=-1, vmax=1, shading='auto', rasterized=True)
    ax.set_xlim(0, 3); ax.set_xlabel(r'distance from wall / $\Lambda_p$'); ax.axhline(0, color=INK2, lw=0.4)
    ax.set_title(f'({"ab"[k]}) {wall} wall', fontsize=8.5, loc='left')
    if k == 0: ax.set_ylabel(r'direction cosine $\mu$')
    else: ax.set_yticklabels([])
    prof[wall] = (d, np.gradient(F['T'], F['zc']), F['zc'])
cb = fig.colorbar(im, cax=fig.add_subplot(gs[2])); cb.ax.tick_params(labelsize=6.5); cb.set_label(r'$\mathrm{Im}\,(g-T)$, normalised', fontsize=7)
ax = fig.add_subplot(gs[4]); d, dTs, zc = prof['specular']; s_ = (d > 2e-3) & (d < 0.3)
gk_c = np.cosh(m * zc) / (np.cosh(m / 2) * (1 + 3 * Lam * L ** 2)) - 1; gk_s = np.cosh(m * zc) / np.cosh(m / 2) - 1
ax.semilogy(d[s_] / Knv, np.abs(gk_s - dTs)[s_], color=C1, label='scalar G-K')
ax.semilogy(d[s_] / Knv, np.abs(gk_c - dTs)[s_], color=C2, ls='--', label='consistent G-K')
ax.semilogy(d[s_] / Knv, np.abs(prof['diffuse'][1] - dTs)[s_], color=INK, lw=1.0, ls=':', label='BTE, diffuse wall')
ax.set_xlim(0, 3); ax.set_ylim(1e-4, 1e-1); ax.set_xlabel(r'distance from wall / $\Lambda_p$'); ax.set_ylabel(r'$|\partial_zT-\partial_zT_{\rm BTE,spec}|$')
ax.set_title('(c) deviation of the gradient', fontsize=8.5, loc='left'); ax.legend(frameon=False, fontsize=6.6, loc='upper right')
ax.grid(True, which='major', color='#e6e5e0', lw=0.5); ax.set_axisbelow(True)
fig.savefig(os.path.join(FIG, 'fig_bte_phase.pdf')); fig.savefig(os.path.join(FIG, 'fig_bte_phase.png'), dpi=160)
