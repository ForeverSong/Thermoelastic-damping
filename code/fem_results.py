"""Finite-element results for the manuscript (Sections 6-7). Writes figures/*.pdf, tables/*.tex and
../fem_results.json. Every number quoted in the FE sections comes from this script."""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gkted as g, fem_mesh as fm, fem_scalar as fs, fem_mixed as mx, fem_mixed_curved as mc
HERE = os.path.dirname(os.path.abspath(__file__))
FIG, TAB = os.path.join(HERE, '..', 'figures'), os.path.join(HERE, '..', 'tables')
C1, C2, C3, C4 = '#257de0', '#f47230', '#00b894', '#f2a600'
GREY, INK, INK2 = '#8c8c8c', '#191919', '#666666'
plt.rcParams.update({'font.size': 8.5, 'axes.labelsize': 8.5, 'legend.fontsize': 7.5, 'axes.edgecolor': INK2,
                     'axes.linewidth': 0.6, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK,
                     'text.color': INK, 'lines.linewidth': 1.5, 'savefig.dpi': 300})
DE, chi, tau = g.SI['DeltaE'], g.SI['chi'], g.SI['tau']

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fem_results.json')
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
save = lambda: json.dump(R, open(OUT, 'w'), indent=1)
Tt = 1e-3j
qrel = lambda a, b: abs(np.imag(a) - np.imag(b)) / abs(np.imag(b))


def ident(r, A):
    return abs(abs(np.imag(r['I'])) - r['D'] / (r['Omega'] * A['zz'])) / abs(np.imag(r['I']))


# ------------------------------------------------------------------ 1. disk convergence
if 'conv' not in R:
    Om, L = 3.39, 0.1; Lam = 1j * Om
    ex = {s: g.disk_I(Lam, L, Tt, s, 'slip') for s in (0.0, L, np.inf)}
    exS = {'S': g.disk_I(Lam, L, Tt, 0, 'S'), 'P': g.disk_I(Lam, L, Tt, 0, 'P'), 'W': g.disk_I_wall(Lam, L, Tt, 0.0)}
    conv = []
    for h in (0.2, 0.1, 0.05, 0.025):
        mesh = fm.disk_mesh(h, grade=(0.4, (L / 4) * (h / 0.2)))
        Sc = mc.build(mesh); Ac = mc.assemble(Sc, L)
        Sp = mx.build(mesh); Ap = mx.assemble(Sp, L)
        row = dict(h=h, nt=int(Sc['nt']), ndof=int(Ac['free'].sum() + Sc['nt']))
        for key, s in (('noslip', 0.0), ('slip', L), ('perfect', np.inf)):
            rc = mc.solve(Ac, Lam, L, Tt, s); rp = mx.solve(Ap, Lam, L, Tt, s)
            row[key] = float(abs(rc['I'] - ex[s]) / abs(ex[s]))
            row[key + '_poly'] = float(abs(rp['I'] - ex[s]) / abs(ex[s]))
            row[key + '_id'] = float(ident(rc, Ac))
        As = fs.assemble(fm.disk_mesh(h))
        for k in ('S', 'P', 'W'):
            I = fs.solve(As, Lam, L, Tt, k, 0.0)[0]
            row['p2_' + k] = float(abs(I - exS[k]) / abs(exS[k]))
        conv.append(row); print(row, flush=True)
    # one more polygonal level to expose the reduced rate
    mesh = fm.disk_mesh(0.0125, grade=(0.4, (L / 4) * (0.0125 / 0.2)))
    Sp = mx.build(mesh); Ap = mx.assemble(Sp, L)
    conv.append(dict(h=0.0125, perfect_poly=float(abs(mx.solve(Ap, Lam, L, Tt, np.inf)['I'] - ex[np.inf]) / abs(ex[np.inf]))))
    R['conv'] = conv; save()

# ------------------------------------------------------------------ 2. robustness in L (fixed meshes)
if 'robust' not in R:
    Om = 3.39; Lam = 1j * Om; rob = []
    for h in (0.2, 0.1, 0.05):
        Sc = mc.build(fm.disk_mesh(h, grade=(0.3, h / 8)))
        for L in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1.0):
            A = mc.assemble(Sc, L)
            for s in (0.0, np.inf):
                ex = g.disk_I(Lam, L, Tt, s, 'slip'); r = mc.solve(A, Lam, L, Tt, s)
                rob.append(dict(h=h, L=L, S='noslip' if s == 0 else 'perfect',
                                err=float(abs(r['I'] - ex) / abs(ex)), idn=float(ident(r, A))))
        print('robust h', h, flush=True)
    R['robust'] = rob; save()

# ------------------------------------------------------------------ 3. silicon rectangular beams (clamped-clamped)
if 'si_rect' not in R:
    rows = []
    for l_nm in (40, 120):
        for h_um in (0.2, 0.5, 1.0, 2.0):
            h, l = h_um * 1e-6, l_nm * 1e-9
            om = g.omega_clamped_rect(h, 1)
            p = g.nondim(om, h, l, tau)
            Lam, L, Ttp = p['Lam'], p['L'], p['Tt']
            pf = g.nondim(om, h, 0.0, 0.0)
            row = dict(l_nm=l_nm, h_um=h_um, Omega=float(np.imag(Lam)), L=L,
                       fourier=DE * abs(np.imag(g.slab_I(pf['Lam'], 0.0, 0.0, 'S'))),
                       S=DE * abs(np.imag(g.slab_I(Lam, L, Ttp, 'S'))),
                       slab=DE * abs(np.imag(g.slab_I(Lam, L, Ttp, 'consistent'))))
            for B in (1, 2, 4):
                mesh = fm.rect_mesh(B, 1.0, 16 * B, 16, wall_h=min(L / 6, 0.02))
                Sx = mx.build(mesh)
                free = lambda x, B=B: 'free' if abs(abs(x[0]) - B / 2) < 1e-9 else 'wall'
                Aw, Af = mx.assemble(Sx, L), mx.assemble(Sx, L, marker=free)
                corr = np.imag(mx.solve(Aw, Lam, L, Ttp, 0.0)['I']) / np.imag(mx.solve(Af, Lam, L, Ttp, 0.0)['I'])
                row[f'rect{B}'] = float(row['slab'] * corr)
                row[f'eff{B}'] = DE * abs(np.imag(g.rect_I_eff(Lam, L, Ttp, B, 0.0)))
            rows.append(row); print(row, flush=True)
    R['si_rect'] = rows; save()

# ------------------------------------------------------------------ tables
c = R['conv']
rate = lambda e1, e0: np.log2(e0 / e1)
with open(os.path.join(TAB, 'tab_fem_conv.tex'), 'w') as f:
    for i, r in enumerate(c[:4]):
        cells = [f"{r['h']:.3f}", f"{r['ndof']}"]
        for k in ('noslip', 'slip', 'perfect'):
            cells.append(f"{r[k]:.2e}")
            cells.append('--' if i == 0 else f"{rate(r[k], c[i-1][k]):.2f}")
        cells.append(f"{r['perfect_poly']:.2e}")
        cells.append('--' if i == 0 else f"{rate(r['perfect_poly'], c[i-1]['perfect_poly']):.2f}")
        cells.append(f"{max(r['noslip_id'], r['slip_id'], r['perfect_id']):.0e}")
        f.write(' & '.join(cells).replace('e-0', 'e-').replace('e-', '$\\times10^{-').replace('$\\times10^{-1', '$\\times10^{-1') + ' \\\\\n')
# simpler, robust formatting of the above (overwrite) -------------------------------------------
def sci(x):
    m, e = f"{x:.2e}".split('e'); return f"${m}\\,(\\mathrm{{{int(e)}}})$"
def sci(x):
    m, e = f"{x:.2e}".split('e'); return f"{m}e{int(e)}"
with open(os.path.join(TAB, 'tab_fem_conv.tex'), 'w') as f:
    for i, r in enumerate(c[:4]):
        cells = [f"{r['h']:.3f}", f"{r['ndof']}"]
        for k in ('noslip', 'slip', 'perfect'):
            cells += [sci(r[k]), '--' if i == 0 else f"{rate(r[k], c[i-1][k]):.2f}"]
        cells += [sci(r['perfect_poly']), '--' if i == 0 else f"{rate(r['perfect_poly'], c[i-1]['perfect_poly']):.2f}"]
        cells += [sci(r['p2_W']), '--' if i == 0 else f"{rate(r['p2_W'], c[i-1]['p2_W']):.2f}"]
        f.write(' & '.join(cells) + ' \\\\\n')
    r = c[4]
    f.write(f"0.0125 & -- & & & & & & & {sci(r['perfect_poly'])} & {rate(r['perfect_poly'], c[3]['perfect_poly']):.2f} & & \\\\\n")
with open(os.path.join(TAB, 'tab_fem_robust.tex'), 'w') as f:
    Ls = sorted({r['L'] for r in R['robust']})
    for h in (0.2, 0.1, 0.05):
        for s in ('noslip', 'perfect'):
            v = {r['L']: r['err'] for r in R['robust'] if r['h'] == h and r['S'] == s}
            f.write(f"{h:.2f} & {'no-slip' if s == 'noslip' else 'perfect slip'} & " + ' & '.join(sci(v[L]) for L in Ls) + ' \\\\\n')
with open(os.path.join(TAB, 'tab_si_rect.tex'), 'w') as f:
    for r in R['si_rect']:
        pc = lambda k: 100 * (r[k] - r['slab']) / r['slab']
        f.write(f"{r['l_nm']} & {r['h_um']:.1f} & {r['L']:.2f} & {r['Omega']:.2f} & {r['S']*1e5:.3f} & {r['slab']*1e5:.3f} & "
                f"{r['rect1']*1e5:.3f} & {pc('rect1'):+.1f} & {pc('rect2'):+.1f} & {pc('rect4'):+.1f} & {pc('eff1'):+.1f} \\\\\n")
R['max_identity'] = float(max([r[k] for r in c[:4] for k in ('noslip_id', 'slip_id', 'perfect_id')] + [r['idn'] for r in R['robust']]))
save()

# ------------------------------------------------------------------ figure: FE convergence + robustness
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.7))
hs = np.array([r['h'] for r in c[:4]])
for k, lab, col, mk in (('noslip', 'no-slip', C2, 'o'), ('slip', r'slip $\ell_s=l$', C3, 's'), ('perfect', 'perfect slip', C4, '^')):
    a1.loglog(hs, [r[k] for r in c[:4]], color=col, marker=mk, ms=4.5, label=f'curved BDM$_1$, {lab}')
hp = np.array([r['h'] for r in c])
a1.loglog(hp, [r['perfect_poly'] for r in c], color=GREY, marker='^', ms=4.5, ls='--', mfc='white', label='polygonal BDM$_1$, perfect slip')
a1.loglog(hs, 0.12 * hs ** 2, color=INK2, lw=0.8, ls=':'); a1.text(0.06, 0.12 * 0.06 ** 2 * 0.42, r'$\propto h^2$', color=INK2, fontsize=7.5)
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
a1.xaxis.set_major_locator(FixedLocator([0.0125, 0.025, 0.05, 0.1, 0.2])); a1.xaxis.set_minor_locator(NullLocator())
a1.xaxis.set_major_formatter(FixedFormatter(['0.0125', '0.025', '0.05', '0.1', '0.2'])); a1.set_ylim(3e-5, 3e-2)
a1.set_xlabel('$h$'); a1.set_ylabel(r'$|I_h-I|/|I|$'); a1.legend(frameon=False, loc='upper left', fontsize=6.4)
a1.set_title('(a) disk, $\\Omega=3.39$, $L=0.1$', fontsize=8.5, loc='left')
Ls = sorted({r['L'] for r in R['robust']})
for h, col in ((0.2, C1), (0.1, C2), (0.05, C3)):
    for s, ls, mk in (('noslip', '-', 'o'), ('perfect', '--', '^')):
        a2.loglog(Ls, [next(r['err'] for r in R['robust'] if r['h'] == h and r['S'] == s and r['L'] == L) for L in Ls],
                  color=col, ls=ls, marker=mk, ms=4, mfc='white' if s == 'perfect' else col,
                  label=f'$h={h}$' if s == 'noslip' else None)
a2.plot([], [], color=INK2, ls='-', marker='o', ms=4, label='no-slip'); a2.plot([], [], color=INK2, ls='--', marker='^', ms=4, mfc='white', label='perfect slip')
a2.set_xlabel('$L=l/a$'); a2.set_ylabel(r'$|I_h-I|/|I|$'); a2.set_ylim(1e-4, 1e-1)
a2.legend(frameon=False, ncol=2, fontsize=6.6, loc='upper center'); a2.set_title('(b) fixed meshes, $L$ varied', fontsize=8.5, loc='left')
for a in (a1, a2):
    a.grid(True, which='major', color='#e6e5e0', lw=0.5); a.set_axisbelow(True)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_fem_conv.pdf')); fig.savefig(os.path.join(FIG, 'fig_fem_conv.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ figure: side-wall correction
d = json.load(open(os.path.join(os.path.dirname(OUT), 'rect_omega.json')))
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=False)
for ax, L in zip(axs, (0.05, 0.1)):
    Om = np.array(sorted({r['Om'] for r in d}))
    for B, col, mk in ((1, C1, 'o'), (2, C2, 's'), (4, C3, '^')):
        rr = [r for r in d if r['B'] == B and r['L'] == L]
        ax.semilogx([r['Om'] for r in rr], [B * r['true'] for r in rr], ls='none', marker=mk, ms=4.5, color=col, mfc='white', mew=1.1,
                    label=f'mixed FE, $b/h={B}$')
    rr = [r for r in d if r['B'] == 4 and r['L'] == L]
    ax.semilogx([r['Om'] for r in rr], [4 * r['W'] for r in rr], color=INK, lw=1.2, label='wall law W ($P_2$), $b/h=4$')
    Of = np.logspace(-0.5, 2.5, 200)
    eff = []
    for o in Of:
        Lam = 1j * o; T_ = 1j * o * 1e-3 * L ** 2; I1 = g.slab_I(Lam, L, T_, 'consistent')
        eff.append(4 * (np.imag(g.rect_I_eff(Lam, L, T_, 4, 0.0)) - np.imag(I1)) / np.imag(I1))
    ax.semilogx(Of, eff, color=GREY, ls='--', lw=1.2, label=r'effective diffusivity $\chi(1-2\delta/b)$')
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_xlabel(r'$\Omega=\omega h^2/\chi$'); ax.set_title(f'({"ab"[L == 0.1]}) $l/h={L}$', fontsize=8.5, loc='left')
    ax.grid(True, which='major', color='#e6e5e0', lw=0.5); ax.set_axisbelow(True)
axs[0].set_ylabel(r'$(b/h)\,\Delta Q^{-1}/Q^{-1}_{\rm slab}$')
axs[0].legend(frameon=False, fontsize=6.4, loc='lower left')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_sidewall.pdf')); fig.savefig(os.path.join(FIG, 'fig_sidewall.png'), dpi=150)
print('max identity residual', R['max_identity'])
