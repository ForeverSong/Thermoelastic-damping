"""Results for the kinetic form (alpha = 1/3, l = mfp/sqrt(5), slip length from Guo & Wang 2018) against the
G-K form (alpha = 2).  Writes ../alpha_results.json, ../tables/tab_alpha_*.tex, ../figures/fig_alpha.*"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import gkted as g

HERE = os.path.dirname(os.path.abspath(__file__)); FIG, TAB = os.path.join(HERE, '..', 'figures'), os.path.join(HERE, '..', 'tables')
C1, C2, C3, C4 = '#257de0', '#f47230', '#00b894', '#f2a600'
GREY, INK, INK2 = '#8c8c8c', '#191919', '#666666'
plt.rcParams.update({'font.size': 8.5, 'axes.labelsize': 8.5, 'legend.fontsize': 7.5, 'axes.edgecolor': INK2,
                     'axes.linewidth': 0.6, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK,
                     'text.color': INK, 'lines.linewidth': 1.5, 'savefig.dpi': 300})
SEQ = LinearSegmentedColormap.from_list('seq', ['#f4f8fd', '#b9d6f5', '#5a9de8', '#1f5fbf', '#0b2e6b'])
A3 = 1.0 / 3.0; R = {}
Q = lambda I: -np.imag(I)


def kin(Om, Kn, wall, alpha=A3):
    """disk, gray kinetic parametrisation: tau = mfp/v, slip length 8/15 mfp for diffuse walls.
    alpha = 1/3: l = mfp/sqrt(5);  alpha = 2: l chosen so that (1+alpha) l^2 is the same (identical slab)."""
    L = Kn / np.sqrt(5) if alpha == A3 else np.sqrt(4.0 / 45.0) * Kn
    Lam, tt = 1j * Om, 1j * Om * Kn ** 2 / 3
    if wall == 'scalar': return g.disk_I(Lam, L, tt, 0.0, 'S', alpha=alpha)
    S = {'noslip': 0.0, 'diffuse': 8.0 * Kn / 15.0, 'specular': np.inf}[wall]
    return g.disk_I(Lam, L, tt, S, 'slip', alpha=alpha)


fourier = lambda Om: g.disk_I(1j * Om, 1e-12, 0.0, 0.0, 'S')
# --- table: kinetic parametrisation at Omega = 3.39
rows = []
with open(os.path.join(TAB, 'tab_alpha_kn.tex'), 'w') as f:
    for Kn in (0.05, 0.1, 0.2, 0.3):
        Om = 3.39; r = dict(Kn=Kn, fourier=Q(fourier(Om)), scalar=Q(kin(Om, Kn, 'scalar')), noslip=Q(kin(Om, Kn, 'noslip')), diffuse=Q(kin(Om, Kn, 'diffuse')),
                            specular=Q(kin(Om, Kn, 'specular')), diffuse_gk=Q(kin(Om, Kn, 'diffuse', 2.0)), noslip_gk=Q(kin(Om, Kn, 'noslip', 2.0)))
        r['scalar_check'] = abs(Q(kin(Om, Kn, 'scalar', 2.0)) - r['scalar'])
        rows.append({k: float(v) for k, v in r.items()})
        f.write(f"{Kn:g} & {r['fourier']:.4f} & {r['scalar']:.4f} & {r['noslip']:.4f} & {r['diffuse']:.4f} & {r['specular']:.4f} & {r['diffuse_gk']:.4f} & "
                f"{100*(r['scalar']-r['diffuse'])/r['diffuse']:+.1f} & {100*(r['diffuse_gk']-r['diffuse'])/r['diffuse']:+.1f} \\\\\n")
R['kn_table'] = rows; print(open(os.path.join(TAB, 'tab_alpha_kn.tex')).read())
# --- silicon cantilevers (mode 2) at fixed l: G-K form vs kinetic form
mat = g.SI; DE, chi, tau = mat['DeltaE'], mat['chi'], mat['tau']; rows = []
with open(os.path.join(TAB, 'tab_alpha_si.tex'), 'w') as f:
    for l_nm in (40, 120):
        for a_um in (0.2, 0.5, 1.0):
            a, l = a_um * 1e-6, l_nm * 1e-9; om = g.omega_cantilever_circ(a, 2); p = g.nondim(om, a, l, tau, l)
            v = {(al, m): DE * Q(g.disk_I(p['Lam'], p['L'], p['Tt'], p['S'] if m == 'slip' else 0.0, m, alpha=al)) for al in (2.0, A3) for m in ('S', 'slip')}
            rows.append(dict(l_nm=l_nm, a_um=a_um, S2=v[(2.0, 'S')], slip2=v[(2.0, 'slip')], S3=v[(A3, 'S')], slip3=v[(A3, 'slip')]))
            f.write(f"{l_nm} & {a_um:.1f} & {v[(2.0,'S')]*1e5:.3f} & {v[(2.0,'slip')]*1e5:.3f} & {100*(v[(2.0,'S')]-v[(2.0,'slip')])/v[(2.0,'slip')]:+.0f} & "
                    f"{v[(A3,'S')]*1e5:.3f} & {v[(A3,'slip')]*1e5:.3f} & {100*(v[(A3,'S')]-v[(A3,'slip')])/v[(A3,'slip')]:+.0f} \\\\\n")
R['si_table'] = [{k: float(v) for k, v in r.items()} for r in rows]; print(open(os.path.join(TAB, 'tab_alpha_si.tex')).read())
# --- largest underestimation by the scalar closure over the scanned range, both forms (slip length l)
Og, Lg = np.logspace(-1, 2, 121), np.logspace(np.log10(0.003), 0, 121)
for al, tag in ((2.0, 'gk'), (A3, 'kin')):
    E = np.array([[(Q(g.disk_I(1j * o, L, 0.0, 0.0, 'S', alpha=al)) - Q(g.disk_I(1j * o, L, 0.0, L, 'slip', alpha=al))) / Q(g.disk_I(1j * o, L, 0.0, L, 'slip', alpha=al)) for o in Og] for L in Lg])
    R[f'scan_{tag}'] = dict(min=float(E.min()), max=float(E.max()), L_10pct=float(Lg[np.argmax((np.abs(E) > 0.1).all(axis=1))]))
    print(tag, R[f'scan_{tag}'])
# --- maps in the (Omega, Kn) plane, kinetic parametrisation
Om = np.logspace(-1, 2, 81); Kn = np.logspace(-2, np.log10(0.5), 61)
D_al = np.zeros((len(Kn), len(Om))); D_sc = np.zeros_like(D_al)
for i, k in enumerate(Kn):
    for j, o in enumerate(Om):
        qd = Q(kin(o, k, 'diffuse')); D_al[i, j] = (Q(kin(o, k, 'diffuse', 2.0)) - qd) / qd; D_sc[i, j] = (Q(kin(o, k, 'scalar')) - qd) / qd
R['map'] = dict(alpha_diff_max=float(np.abs(D_al).max()), alpha_diff_at_Kn01_max=float(np.abs(D_al[np.argmin(abs(Kn - 0.1))]).max()),
                scalar_min=float(D_sc.min()), scalar_max=float(D_sc.max()))
print(R['map']); json.dump(R, open(os.path.join(HERE, '..', 'alpha_results.json'), 'w'), indent=1)
fig = plt.figure(figsize=(6.9, 2.6)); gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.3, 1, 0.36, 1], wspace=0.08, left=0.075, right=0.935, top=0.88, bottom=0.18)
ax = fig.add_subplot(gs[0]); Of = np.logspace(-1, 2.3, 160); K = 0.2
ax.loglog(Of, [Q(fourier(o)) for o in Of], color=GREY, lw=1.0, ls=(0, (4, 2)), label='Fourier')
ax.loglog(Of, [Q(kin(o, K, 'scalar')) for o in Of], color=C1, label='scalar closure')
qd, qs = np.array([Q(kin(o, K, 'diffuse')) for o in Of]), np.array([Q(kin(o, K, 'specular')) for o in Of])
ax.fill_between(Of, qd, qs, color=C2, alpha=0.3, lw=0); ax.loglog(Of, qd, color=C2, ls='--', label='consistent, diffuse wall'); ax.loglog(Of, qs, color=C2, lw=0.9, label='consistent, specular wall')
ax.loglog(Of, [Q(kin(o, K, 'diffuse', 2.0)) for o in Of], color=C3, lw=1.0, ls='-.', label=r'G-K form ($\alpha=2$), diffuse')
ax.set_ylim(1.2e-4, 1.5); ax.set_xlabel(r'$\Omega=\omega a^2/\chi$'); ax.set_ylabel(r'$Q^{-1}/\Delta_E$'); ax.set_title('(a) disk, Kn $=0.2$', fontsize=8.5, loc='left')
ax.legend(frameon=False, fontsize=6.2, loc='lower left', handlelength=1.9); ax.grid(True, which='major', color='#e6e5e0', lw=0.5); ax.set_axisbelow(True)
from matplotlib.colors import LogNorm
ax = fig.add_subplot(gs[2]); D = 100 * np.abs(D_al)
im = ax.pcolormesh(Om, Kn, D, cmap=SEQ, vmin=0, vmax=np.nanmax(D), shading='auto', rasterized=True)
cs = ax.contour(Om, Kn, D, levels=[0.5, 2], colors=[INK], linewidths=[0.6, 1.0], linestyles=['--', '-']); ax.clabel(cs, fmt='%g%%', fontsize=6.5)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$\Omega$'); ax.set_ylabel('Kn', labelpad=1); ax.set_title(r'(b) effect of $\alpha$ on $Q^{-1}$ (%)', fontsize=8.5, loc='left')
cb = fig.colorbar(im, cax=ax.inset_axes([1.03, 0, 0.045, 1])); cb.ax.tick_params(labelsize=6.5)
ax = fig.add_subplot(gs[4]); D = np.clip(np.abs(D_sc), 1e-3, None)
im = ax.pcolormesh(Om, Kn, D, cmap=SEQ, norm=LogNorm(1e-3, 30), shading='auto', rasterized=True)
cs = ax.contour(Om, Kn, D, levels=[1e-2, 1e-1, 1.0], colors=[INK], linewidths=[0.6, 0.8, 1.1], linestyles=[':', '--', '-']); ax.clabel(cs, fmt={1e-2: '1%', 1e-1: '10%', 1.0: '100%'}, fontsize=6.5)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$\Omega$'); ax.set_title('(c) error of the scalar closure', fontsize=8.5, loc='left')
cb = fig.colorbar(im, cax=ax.inset_axes([1.03, 0, 0.045, 1])); cb.ax.tick_params(labelsize=6.5)
fig.savefig(os.path.join(FIG, 'fig_alpha.pdf')); fig.savefig(os.path.join(FIG, 'fig_alpha.png'), dpi=160)
