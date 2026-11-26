"""Kinetic validation (manuscript Section 8): gray phonon Boltzmann equation vs the continuum closures in the slab.
Writes ../bte_results.json, ../tables/tab_bte.tex, ../figures/fig_bte.{pdf,png}."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import bte1d as b

HERE = os.path.dirname(os.path.abspath(__file__))
FIG, TAB = os.path.join(HERE, '..', 'figures'), os.path.join(HERE, '..', 'tables')
C1, C2, C3, C4 = '#257de0', '#f47230', '#00b894', '#f2a600'
GREY, INK, INK2 = '#8c8c8c', '#191919', '#666666'
plt.rcParams.update({'font.size': 8.5, 'axes.labelsize': 8.5, 'legend.fontsize': 7.5, 'axes.edgecolor': INK2,
                     'axes.linewidth': 0.6, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK,
                     'text.color': INK, 'lines.linewidth': 1.5, 'savefig.dpi': 300})
R = {}


def dom(Om, Kn, wall):
    """Richardson-extrapolated discrete-ordinates value (scheme is second order in the cell size)."""
    a, c = b.bte_dom(Om, Kn, wall, N=600, Nmu=32), b.bte_dom(Om, Kn, wall, N=1200, Nmu=32)
    return c + (c - a) / 3.0


# 0. accuracy of the extrapolated discrete-ordinates solver (specular walls, where the mode sum is exact)
chk = [abs(dom(Om, Kn, 'specular') - b.bte_specular(Om, Kn)) / abs(1 + b.bte_specular(Om, Kn))
       for Om, Kn in ((1.0, 0.05), (10.0, 0.1), (10.0, 0.5), (100.0, 0.2), (3.0, 1.0), (0.3, 0.2))]
R['dom_check_max'] = float(max(chk)); print('DOM (extrapolated) vs mode sum, max error:', max(chk))

# 1. low-frequency limit: -Im I / Omega  (Fourier and scalar closure: 1/10;  consistent G-K: 1/10 + 4 Kn^2/15)
low = []
for Kn in (0.05, 0.1, 0.2, 0.4):
    Om = 1e-2; c = b.closures(Om, Kn)
    low.append(dict(Kn=Kn, bte_spec=float(-np.imag(b.bte_specular(Om, Kn)) / Om), bte_diff=float(-np.imag(dom(Om, Kn, 'diffuse')) / Om),
                    **{k: float(-np.imag(v) / Om) for k, v in c.items()}))
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in low[-1].items()})
R['low'] = low

# 2. error of each closure against the kinetic solution as Kn -> 0, measured in |I - I_BTE| / |1 + I_BTE|
err = []
for Om in (3.0, 10.0, 30.0):
    for Kn in (0.4, 0.2, 0.1, 0.05, 0.025):
        ref = b.bte_specular(Om, Kn); dif = dom(Om, Kn, 'diffuse'); c = b.closures(Om, Kn)
        e = lambda I, r=ref: float(abs(I - r) / abs(1 + r))
        err.append(dict(Om=Om, Kn=Kn, **{k: e(v) for k, v in c.items()}, diffuse_vs_specular=e(dif),
                        consistent_vs_diffuse=e(c['consistent'], dif), scalar_vs_diffuse=e(c['scalar'], dif)))
        print({k: (f"{v:.2e}" if k not in ('Om', 'Kn') else v) for k, v in err[-1].items()}, flush=True)
R['err'] = err

# 3. damping curves
Oms = np.logspace(-1, 3, 33); curves = {}
for Kn in (0.1, 0.3):
    rows = []
    for Om in Oms:
        c = b.closures(Om, Kn)
        rows.append(dict(Om=float(Om), bte_spec=float(-np.imag(b.bte_specular(Om, Kn))), bte_diff=float(-np.imag(dom(Om, Kn, 'diffuse'))),
                         **{k: float(-np.imag(v)) for k, v in c.items()}))
    curves[str(Kn)] = rows
    pk = lambda key: max(r[key] for r in rows)
    print(f"Kn={Kn}: peak -Im I  BTE spec {pk('bte_spec'):.4f}  diffuse {pk('bte_diff'):.4f}  consistent {pk('consistent'):.4f}  scalar {pk('scalar'):.4f}  Fourier {pk('fourier'):.4f}")
R['curves'] = curves
# 4. temperature gradient near the wall: kinetic solution vs the two G-K wall values (Omega = 10, Kn = 0.1)
Om, Kn = 10.0, 0.1; Lam, L, tt = b.gk_params(Om, Kn)
F_ = b.bte_dom(Om, Kn, 'specular', N=1200, Nmu=32, return_fields=True); zc, T, xe, q = F_['zc'], F_['T'], F_['xe'], F_['q']
dT = np.gradient(T, zc); m = np.sqrt(Lam * (1 + tt) / (1 + 3 * Lam * L ** 2))
gk = lambda z: np.cosh(m * z) / (np.cosh(m / 2) * (1 + 3 * Lam * L ** 2)) - 1
R['wall_gradient'] = dict(Om=Om, Kn=Kn, rows=[dict(dist=float(d), bte=[float(np.interp(0.5 - d, zc, dT.real)), float(np.interp(0.5 - d, zc, dT.imag))],
                          gk=[float(gk(0.5 - d).real), float(gk(0.5 - d).imag)]) for d in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3)])
for r in R['wall_gradient']['rows']:
    print(f"  distance {r['dist']:.2f}: dT/dz kinetic {r['bte'][0]:+.4f}{r['bte'][1]:+.4f}i   consistent G-K {r['gk'][0]:+.4f}{r['gk'][1]:+.4f}i")
print('  |q| at the wall (kinetic):', abs(q[-1]))
json.dump(R, open(os.path.join(HERE, '..', 'bte_results.json'), 'w'), indent=1)

# table: errors at Omega = 3 and 30
sci = lambda x: f"{x:.1e}".replace('e-0', 'e-').replace('e+0', 'e')
with open(os.path.join(TAB, 'tab_bte.tex'), 'w') as f:
    for Om in (3.0, 30.0):
        for r in [r for r in err if r['Om'] == Om]:
            f.write(f"{Om:g} & {r['Kn']:g} & {sci(r['fourier'])} & {sci(r['cv'])} & {sci(r['scalar'])} & {sci(r['consistent'])} & "
                    f"{sci(r['diffuse_vs_specular'])} & {sci(r['scalar_vs_diffuse'])} & {sci(r['consistent_vs_diffuse'])} \\\\\n")

