"""Observed convergence rates against the a priori estimates (affine meshes, exact slab solution in the unit square,
side walls free, top and bottom walls no-slip).  Theory: energy norm of the flux error O(h), ||T - T_h|| O(h),
damping output I_h O(h^2).  Writes ../tables/tab_rates.tex, ../rates_check.json"""
import numpy as np, json, os
import gkted as g, fem_mesh as fm, fem_mixed as mx
Tt = 1e-3j; Lam, L = 10j, 0.1; out = {}
marker = lambda x: 'free' if abs(abs(x[0]) - 0.5) < 1e-9 else 'wall'


def exact(alpha):
    c = 1 + alpha; a = 1 + c * Lam * L ** 2; m = np.sqrt(Lam * (1 + Tt) / a); b = 1.0 / (a * m * np.cosh(m / 2))
    T = lambda z: b * np.sinh(m * z) - z
    qz = lambda z: -Lam * b * (np.cosh(m * z) - np.cosh(m / 2)) / m
    dv = lambda z: -Lam * b * np.sinh(m * z)
    dqz = dv                                             # d q_z / dz = div q
    return T, qz, dv


def errors(n, alpha):
    S = mx.build(fm.rect_mesh(1.0, 1.0, n, n)); A = mx.assemble(S, L, marker=marker); r = mx.solve(A, Lam, L, Tt, 0.0, alpha=alpha)
    T, qz, dv = exact(alpha); eq = ed = eT = eg = 0.0
    for k, el in enumerate(S['els']):
        X = el['X']; pts = 0.5 * (X[[1, 2, 0]] + X[[2, 0, 1]]); w = el['area'] / 3.0      # edge-midpoint rule
        qd = r['q'][el['gd']]; qh = np.einsum('qic,i->qc', mx._trace(el, pts), qd)
        gh = np.einsum('iab,i->ab', el['grad'], qd)                                       # grad q_h (constant)
        z = pts[:, 1]
        eq += w * np.sum(np.abs(qh[:, 0]) ** 2 + np.abs(qh[:, 1] - qz(z)) ** 2)
        ed += w * np.sum(np.abs(el['div'] @ qd - dv(z)) ** 2)
        eT += w * np.sum(np.abs(r['T'][k] - T(z)) ** 2)
        eg += w * np.sum(np.abs(gh[0, 0]) ** 2 + np.abs(gh[0, 1]) ** 2 + np.abs(gh[1, 0]) ** 2 + np.abs(gh[1, 1] - dv(z)) ** 2)
    Iex = g.slab_I(Lam, L, Tt, 'consistent', alpha=alpha)
    return dict(n=n, q=float(np.sqrt(eq)), grad=float(np.sqrt(eg)), div=float(np.sqrt(ed)), T=float(np.sqrt(eT)), I=float(abs(r['I'] - Iex)))


rate = lambda a, b_: np.log2(a / b_)
sci = lambda x: f"{x:.2e}".replace('e-0', 'e-')
with open(os.path.join('..', 'tables', 'tab_rates.tex'), 'w') as f:
    for alpha, lab in ((2.0, '2'), (1.0 / 3.0, '1/3')):
        rows = [errors(n, alpha) for n in (8, 16, 32, 64)]; out[lab] = rows
        for i, r in enumerate(rows):
            cells = [f"${lab}$" if i == 0 else '', f"1/{r['n']}"]
            for k in ('q', 'grad', 'div', 'T', 'I'):
                cells += [sci(r[k]), '--' if i == 0 else f"{rate(rows[i-1][k], r[k]):.2f}"]
            f.write(' & '.join(cells) + ' \\\\\n'); print(' | '.join(cells), flush=True)
json.dump(out, open('../rates_check.json', 'w'), indent=1)
