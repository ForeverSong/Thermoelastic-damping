"""Figures and tables for the Paper A draft. All numbers quoted in the manuscript are
generated here (tables/*.tex, results.json) to avoid transcription errors."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import gkted as g

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, '..', 'figures'); os.makedirs(FIG, exist_ok=True)
TAB = os.path.join(HERE, '..', 'tables'); os.makedirs(TAB, exist_ok=True)

# validated categorical slots (fixed order) + neutral reference
C1, C2, C3, C4 = '#257de0', '#f47230', '#00b894', '#f2a600'
GREY, INK, INK2 = '#8c8c8c', '#191919', '#666666'
plt.rcParams.update({'font.size': 8.5, 'axes.labelsize': 8.5, 'legend.fontsize': 7.5,
                     'axes.edgecolor': INK2, 'axes.linewidth': 0.6, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.labelcolor': INK, 'text.color': INK,
                     'lines.linewidth': 1.5, 'savefig.dpi': 300})
mat = g.SI
DE, chi, tau = mat['DeltaE'], mat['chi'], mat['tau']

MODELS = [  # (key, label, color, linestyle)
    ('S', 'scalar G-K, $\\partial_n T=0$ (literature)', C1, '-'),
    ('noslip', 'consistent, no-slip ($\\ell_s=0$)', C2, '--'),
    ('slip', 'consistent, slip ($\\ell_s=l$)', C3, '-.'),
    ('perfect', 'consistent, perfect slip ($\\ell_s\\to\\infty$)', C4, ':'),
]


def qinv_disk(a, l, mode, key, L_over_a=10.0):
    om = g.omega_cantilever_circ(a, mode, L_over_a)
    if key == 'fourier':
        p = g.nondim(om, a, 0.0, 0.0)
        return DE * abs(np.imag(g.disk_I(p['Lam'], 1e-12, 0.0, 0.0, 'S')))
    ls = {'S': 0.0, 'noslip': 0.0, 'slip': l, 'perfect': np.inf}[key]
    p = g.nondim(om, a, l, tau, ls)
    model = 'S' if key == 'S' else 'slip'
    return DE * abs(np.imag(g.disk_I(p['Lam'], p['L'], p['Tt'], p['S'], model)))


def qinv_slab(h, l, key, mode=1):
    om = g.omega_clamped_rect(h, mode)
    if key == 'fourier':
        p = g.nondim(om, h, 0.0, 0.0)
        return DE * abs(np.imag(g.slab_I(p['Lam'], 0.0, 0.0, 'S')))
    p = g.nondim(om, h, l, tau)
    return DE * abs(np.imag(g.slab_I(p['Lam'], p['L'], p['Tt'], 'S' if key == 'S' else 'consistent')))


results = {'DeltaE': DE, 'chi': chi, 'tau': tau}

# ---------------------------------------------------------------- Fig. 1: disk
a_um = np.logspace(np.log10(0.05), np.log10(20), 400)
fig, axs = plt.subplots(2, 2, figsize=(6.6, 4.9), sharex=True)
for i, mode in enumerate((1, 2)):
    for j, l_nm in enumerate((40, 120)):
        ax = axs[i, j]
        l = l_nm * 1e-9
        qf = [qinv_disk(a * 1e-6, l, mode, 'fourier') for a in a_um]
        ax.plot(a_um, qf, color=GREY, lw=1.0, ls=(0, (4, 2)), label='Fourier (reference)')
        for key, lab, col, ls in MODELS:
            q = [qinv_disk(a * 1e-6, l, mode, key) for a in a_um]
            ax.plot(a_um, q, color=col, ls=ls, label=lab)
        ax.axvspan(a_um[0], 2 * l_nm * 1e-3, color='#f0efec', zorder=0, lw=0)
        ax.text(1.15 * a_um[0], 4.5e-7, '$l>a/2$', fontsize=7, color=INK2)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_ylim(3e-7, 2.5e-4)
        ax.grid(True, which='major', color='#e6e6e6', lw=0.5)
        ax.set_title(f'mode {mode},  $l$ = {l_nm} nm', fontsize=8.5, color=INK)
        if i == 1: ax.set_xlabel('radius $a$ ($\\mu$m)')
        if j == 0: ax.set_ylabel('$Q^{-1}$')
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
h, lab = axs[0, 0].get_legend_handles_labels()
fig.legend(h, lab, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(os.path.join(FIG, 'fig_disk_qinv.pdf')); fig.savefig(os.path.join(FIG, 'fig_disk_qinv.png'))
plt.close(fig)

# ---------------------------------------------------------------- Table: disk numbers
rows = []
for mode in (1, 2):
    for l_nm in (40, 120):
        for a in (0.2, 0.5, 1.0, 2.0):
            l = l_nm * 1e-9
            vals = {k: qinv_disk(a * 1e-6, l, mode, k) for k in ('fourier', 'S', 'noslip', 'slip', 'perfect')}
            om = g.omega_cantilever_circ(a * 1e-6, mode)
            rows.append(dict(mode=mode, l_nm=l_nm, a_um=a, Omega=om * (a * 1e-6) ** 2 / chi, **vals))
results['disk_table'] = rows
with open(os.path.join(TAB, 'tab_disk.tex'), 'w') as f:
    for r in rows:
        rel = lambda k: 100 * (r[k] - r['S']) / r['S']
        f.write(f"{r['mode']} & {r['l_nm']} & {r['a_um']:.1f} & {r['Omega']:.2f} & "
                f"{r['S']*1e5:.3f} & {r['noslip']*1e5:.3f} & {r['slip']*1e5:.3f} & {r['perfect']*1e5:.3f} & "
                f"{int(round(rel('noslip'))):+d} & {int(round(rel('slip'))):+d} & {int(round(rel('perfect'))):+d} \\\\\n")

# ---------------------------------------------------------------- Fig. 2: error map
Om = np.logspace(-1, 3, 240)
Lg = np.logspace(-3, 0, 200)
E = np.zeros((Lg.size, Om.size))
for ii, L in enumerate(Lg):
    for jj, o in enumerate(Om):
        Lam = 1j * o
        qs = abs(np.imag(g.disk_I(Lam, L, 0.0, 0.0, 'S')))
        qr = abs(np.imag(g.disk_I(Lam, L, 0.0, L, 'slip')))
        E[ii, jj] = (qs - qr) / qr
results['map_sign'] = dict(min=float(E.min()), max=float(E.max()))
fig, ax = plt.subplots(figsize=(4.8, 3.7))
from matplotlib.colors import BoundaryNorm, ListedColormap
levels = [0.0, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
cmap = ListedColormap(plt.get_cmap('Blues')(np.linspace(0.18, 0.92, len(levels) - 1)))
cf = ax.contourf(Om, Lg, np.clip(np.abs(E), 0, 0.999), levels=levels, cmap=cmap,
                 norm=BoundaryNorm(levels, cmap.N))
ax.contour(Om, Lg, np.abs(E), levels=levels[1:-1], colors='white', linewidths=0.5)
traj = [(1, 40, '-'), (1, 120, '--'), (2, 40, '-'), (2, 120, '--')]
for mode, l_nm, ls in traj:
    a = np.logspace(np.log10(0.05e-6), np.log10(20e-6), 200)
    om = g.omega_cantilever_circ(a, mode)
    x, y = om * a ** 2 / chi, l_nm * 1e-9 / a
    col = C2 if mode == 1 else '#ee7fac'
    ax.plot(x, y, color=col, ls=ls, lw=1.3, label=f'Si cantilever, mode {mode}, $l$ = {l_nm} nm')
ax.set_xscale('log'); ax.set_yscale('log')
ax.legend(loc='upper right', fontsize=6.3, frameon=True, framealpha=0.95, edgecolor='none')
ax.set_xlim(Om[0], Om[-1]); ax.set_ylim(Lg[0], Lg[-1])
ax.set_xlabel('$\\Omega=\\omega a^2/\\chi$'); ax.set_ylabel('$l/a$')
cb = fig.colorbar(cf, ax=ax, ticks=levels)
cb.ax.set_yticklabels(['0', '1%', '3%', '10%', '30%', '50%', '100%'])
cb.set_label('$|Q^{-1}_{\\rm scalar}-Q^{-1}_{\\rm slip}|\\,/\\,Q^{-1}_{\\rm slip}$')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_error_map.pdf')); fig.savefig(os.path.join(FIG, 'fig_error_map.png'))
plt.close(fig)

# ---------------------------------------------------------------- Fig. 3: slab
h_um = np.logspace(np.log10(0.05), np.log10(20), 400)
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
for j, l_nm in enumerate((40, 120)):
    ax = axs[j]; l = l_nm * 1e-9
    ax.plot(h_um, [qinv_slab(h * 1e-6, l, 'fourier') for h in h_um], color=GREY, lw=1.0, ls=(0, (4, 2)), label='Fourier (reference)')
    ax.plot(h_um, [qinv_slab(h * 1e-6, l, 'S') for h in h_um], color=C1, ls='-', label='scalar G-K, $\\partial_z T=0$')
    ax.plot(h_um, [qinv_slab(h * 1e-6, l, 'cons') for h in h_um], color=C2, ls='--', label='consistent, $q(\\pm h/2)=0$')
    hv = np.array([0.1, 0.3, 1.0, 3.0, 10.0])
    fd = []
    for h in hv:
        om = g.omega_clamped_rect(h * 1e-6); p = g.nondim(om, h * 1e-6, l, tau)
        fd.append(DE * abs(np.imag(g.slab_fd(p['Lam'], p['L'], p['Tt'], N=2000))))
    ax.plot(hv, fd, 'o', mfc='white', mec=INK, ms=4.5, mew=0.9, label='finite differences, mixed $(q,T)$')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-6, 2.5e-4)
    ax.grid(True, which='major', color='#e6e6e6', lw=0.5)
    ax.set_title(f'$l$ = {l_nm} nm', fontsize=8.5); ax.set_xlabel('thickness $h$ ($\\mu$m)')
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
axs[0].set_ylabel('$Q^{-1}$')
axs[1].legend(frameon=False, loc='lower right')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_slab_qinv.pdf')); fig.savefig(os.path.join(FIG, 'fig_slab_qinv.png'))
plt.close(fig)

slab_rows = []
for l_nm in (40, 120):
    for h in (0.2, 0.5, 1.0, 2.0):
        l = l_nm * 1e-9
        om = g.omega_clamped_rect(h * 1e-6)
        s, c = qinv_slab(h * 1e-6, l, 'S'), qinv_slab(h * 1e-6, l, 'cons')
        slab_rows.append(dict(l_nm=l_nm, h_um=h, eps=3 * om * l ** 2 / chi, S=s, cons=c))
results['slab_table'] = slab_rows
with open(os.path.join(TAB, 'tab_slab.tex'), 'w') as f:
    for r in slab_rows:
        f.write(f"{r['l_nm']} & {r['h_um']:.1f} & {r['eps']:.3f} & {r['S']*1e5:.3f} & {r['cons']*1e5:.3f} & "
                f"{int(round(100*(r['cons']-r['S'])/r['S'])):+d} \\\\\n")

# ---------------------------------------------------------------- Fig. 4: asymptotics
# colour = wall condition, line style = order of the formula (Appendix A), markers = exact solution
from matplotlib.lines import Line2D
Lv = np.logspace(-3, -0.5, 30)
Omf, Tt = 3.39, 0.0
beta = 1.0
fig, ax = plt.subplots(figsize=(4.6, 3.2))
WALLS = ((0.0, C2, 'o', 'no-slip'), (1.0, C3, 's', 'slip $\\ell_s=l$'), (np.inf, C4, '^', 'perfect slip'))
for Cs, col, mk, lab in WALLS:
    ex, fo, so = [], [], []
    for L in Lv:
        m = np.sqrt(1j * Omf / (1 + 3j * Omf * L ** 2)); P = m * g.rho1(m)
        Is = g.disk_I(1j * Omf, L, Tt, np.inf if np.isinf(Cs) else Cs * L, 'slip'); Ip = g.disk_I(1j * Omf, L, Tt, 0, 'P')
        ex.append(abs((Is + 1) / (Ip + 1) - 1))
        if np.isinf(Cs):                                   # first-order term vanishes
            fo.append(np.nan); so.append(abs(2 * (1 - 1 / P) * L ** 2 / beta ** 2))
        else:
            c = Cs * beta; d = L / (beta * (1 + c))
            fo.append(abs(d * (1 - 1 / P)))
            so.append(abs(d * (1 - 1 / P) * (1 + (L / beta) * (0.5 + 1 / P - 2 * c ** 2) / (1 + c))))
    ax.plot(Lv, fo, color=col, lw=1.2)
    ax.plot(Lv, so, color=col, lw=1.2, ls=(0, (4, 2)))
    ax.plot(Lv[::3], np.array(ex)[::3], mk, color=col, mfc='white', ms=4.5, mew=0.9)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$l/a$'); ax.set_ylabel('$|(I+1)/(I_P+1)-1|$')
ax.grid(True, which='major', color='#e6e6e6', lw=0.5)
for sp_ in ('top', 'right'): ax.spines[sp_].set_visible(False)
leg1 = ax.legend(handles=[Line2D([], [], color=col, marker=mk, mfc='white', ms=4.5, mew=0.9, lw=1.2, label=lab) for _, col, mk, lab in WALLS],
                 frameon=False, fontsize=7, loc='upper left', handlelength=2.2)
ax.add_artist(leg1)
ax.legend(handles=[Line2D([], [], color=INK2, marker='o', mfc='white', ms=4.5, mew=0.9, lw=0, label='exact'),
                   Line2D([], [], color=INK2, lw=1.2, label='first order'),
                   Line2D([], [], color=INK2, lw=1.2, ls=(0, (4, 2)), label='second order')],
          frameon=False, fontsize=7, loc='lower right', handlelength=2.6)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_asymptotics.pdf')); fig.savefig(os.path.join(FIG, 'fig_asymptotics.png'))
plt.close(fig)

# ---------------------------------------------------------------- verification table
ver = []
for Om_, L, S in [(3.4, 0.1, 0.0), (3.4, 0.1, 0.1), (3.4, 0.1, np.inf), (20.0, 0.3, 0.3), (100.0, 0.02, 0.0)]:
    Lam = 1j * Om_
    Ic = g.disk_I(Lam, L, 1e-3j, S, 'slip')
    errs = [abs(Ic - g.disk_collocation(Lam, L, 1e-3j, S, 'slip', N=N)) / abs(Ic) for N in (31, 61, 121)]
    en = g.disk_energy_check(Lam, L, 1e-3j, S, 'slip')
    ver.append(dict(Omega=Om_, L=L, S=('\\infty' if np.isinf(S) else f'{S:.2f}'), I=Ic, errs=errs,
                    eid=abs(en['moment'] - en['dissipation']) / en['moment']))
with open(os.path.join(TAB, 'tab_verify.tex'), 'w') as f:
    for r in ver:
        I = r['I']
        f.write(f"{r['Omega']:.1f} & {r['L']:.2f} & ${r['S']}$ & ${I.real:.8f}{I.imag:+.8f}\\,i$ & "
                + ' & '.join(f"\\num{{{e:.1e}}}" for e in r['errs']) + f" & \\num{{{r['eid']:.1e}}} \\\\\n")

with open(os.path.join(HERE, '..', 'results.json'), 'w') as f:
    json.dump(results, f, indent=1, default=lambda o: [o.real, o.imag] if isinstance(o, complex) else str(o))
print('map relative error range (S vs slip):', results['map_sign'])
for r in rows: print(r)
for r in slab_rows: print(r)
