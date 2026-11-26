"""Numerical check of the coercivity step in the quasi-optimality proof (affine meshes).
For H = Hermitian part of (1 + i gamma) B_h and N = matrix of the norm |||.|||_h the proof gives
    lambda_min(H, N) >= min{1/2, 1 - 2 eta0/eta, gamma},   gamma = min{1, 1/(2 omega tau)},
with eta0 = max{3, 3(1 + rho)/4}.  Writes ../coercivity_check.json"""
import numpy as np, json, itertools
from scipy.linalg import eigh
import fem_mesh as fm, fem_mixed as mx

ETA = 10.0


def rho_of(mesh):
    p, t = mesh['p'], mesh['t']
    u, v = p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]]
    ar = 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    own = {}
    for k, tri in enumerate(t[:, :3]):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            own.setdefault((min(tri[a], tri[b]), max(tri[a], tri[b])), []).append(k)
    r = [max(ar[v[0]] / ar[v[1]], ar[v[1]] / ar[v[0]]) for v in own.values() if len(v) == 2]
    return float(max(r))


def check(name, mesh, marker=None):
    S = mx.build(mesh); rho = rho_of(mesh); eta0 = max(3.0, 0.75 * (1 + rho)); worst = np.inf; rows = []
    A1 = mx.assemble(S, 1.0, eta=ETA, marker=marker); A0 = mx.assemble(S, 1.0, eta=0.0, marker=marker)
    f = A1['free']; d = lambda M: M.toarray()[np.ix_(f, f)]
    Mq, Vq, Vd, Jq, Bt, Bn = (d(A1[k]) for k in ('Mq', 'Vq', 'Vd', 'Jq', 'Bt', 'Bn'))
    Pen, PenB = Jq - d(A0['Jq']), Bn - d(A0['Bn'])              # eta * penalty matrices
    dev2 = Vq - 2 * Vd                                           # 2 (dev eps, dev eps)
    for L, Om, wt, Ss, al in itertools.product((1e-3, 0.1, 1.0), (0.1, 10.0, 1e3), (0.0, 0.3, 5.0), (0.0, 0.5, np.inf), (0.0, 1 / 3, 2.0)):
        B = (1 + 1j * wt) * Mq + L ** 2 * (Vq + (al - 2) * Vd + Jq) + Vd / (1j * Om)
        N = Mq + L ** 2 * (dev2 + al * Vd + Pen) + Vd / Om
        if Ss == 0: B = B + L ** 2 * Bn; N = N + L ** 2 * PenB
        elif np.isfinite(Ss): B = B + (L ** 2 / Ss) * Bt; N = N + (L ** 2 / Ss) * Bt
        g = min(1.0, 1 / (2 * wt)) if wt > 0 else 1.0
        G = (1 + 1j * g) * B; H = 0.5 * (G + G.conj().T)
        lam = eigh(H, N, eigvals_only=True, subset_by_index=[0, 0])[0]
        bound = min(0.5, 1 - 2 * eta0 / ETA, g); worst = min(worst, lam - bound)
        rows.append((L, Om, wt, Ss, al, float(lam), bound))
    lo = min(rows, key=lambda r: r[5])
    print(f"{name:26s} dofs={int(f.sum()):5d} rho={rho:5.2f} eta0={eta0:4.2f} bound(gamma=1)={min(0.5, 1 - 2 * eta0 / ETA):5.2f}  "
          f"min lambda={lo[5]:.4f} at (L,Om,wt,S,al)={lo[:5]}  min(lambda-bound)={worst:+.4f}  cases={len(rows)}", flush=True)
    return dict(dofs=int(f.sum()), rho=rho, eta0=eta0, min_lambda=lo[5], argmin=[float(x) for x in lo[:5]], min_margin=float(worst), cases=len(rows))


out = {}
out['square 6x6'] = check('square 6x6', fm.rect_mesh(1.0, 1.0, 6, 6))
out['rectangle 4:1, 12x4'] = check('rectangle 4:1, 12x4', fm.rect_mesh(4.0, 1.0, 12, 4))
out['square graded'] = check('square graded', fm.rect_mesh(1.0, 1.0, 8, 8, wall_h=0.02))
out['polygonal disk h=0.3'] = check('polygonal disk h=0.3', fm.disk_mesh(0.3))
json.dump(out, open('../coercivity_check.json', 'w'), indent=1)
