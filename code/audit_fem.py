"""Audit part 2: positivity of the interior-penalty form; attribution of the O(delta^2/b) term; wall-model resonance."""
import numpy as np, scipy.sparse as sp
import fem_mesh as fm, fem_mixed as mx, fem_mixed_curved as mc, fem_scalar as fs, gkted as g

print("== 5. min eigenvalue of the viscous form 2(eps,eps)+(div,div)+j_h on V_h (eta=10); must be >= 0")
def mineig(A, noslip):
    f = A['free']
    M = (A['Vq'] + A['Jq'] + (A['Bn'] if noslip else 0 * A['Bn']))[f][:, f].toarray()
    M = 0.5 * (M + M.T)
    asym = abs((A['Vq'] + A['Jq'] + A['Bn']) - (A['Vq'] + A['Jq'] + A['Bn']).T).max()
    w = np.linalg.eigvalsh(M)
    return w[0], w[1], w[-1], asym
for name, mesh, mod in (("straight disk h=0.2 graded", fm.disk_mesh(0.2, grade=(0.4, 0.025)), mx),
                        ("curved   disk h=0.2 graded", fm.disk_mesh(0.2, grade=(0.4, 0.025)), mc),
                        ("curved   disk h=0.1, h_w=h/8", fm.disk_mesh(0.1, grade=(0.3, 0.0125)), mc),
                        ("rectangle 2x1, wall_h=0.005", fm.rect_mesh(2, 1.0, 16, 8, wall_h=0.005), mx)):
    S = mod.build(mesh); A = mod.assemble(S, 0.1)
    for ns in (False, True):
        lo, lo2, hi, asym = mineig(A, ns)
        print(f"  {name:30s} {'no-slip (Nitsche)' if ns else 'perfect slip     '}: lambda_min={lo:+.3e} second={lo2:.3e} lambda_max={hi:.2e}  asym={asym:.0e}")

print("== 6. rectangle: which part of the wall model produces the O(delta^2/b) excess over the effective-diffusivity formula?")
def split_boundary(A, pred):
    m = A['mesh']; X, dofs = m['X'], m['dofs']; nd = len(X)
    loc = [(1, 2, 3), (2, 0, 4), (0, 1, 5)]
    r, c, v = [], [], []; Bsz = np.zeros(nd)
    for k in range(len(dofs)):
        for j in range(3):
            if m['bnd'][m['t2e'][k, j]]:
                a, b_, cc = loc[j]; ed = dofs[k][[a, cc, b_]]; Xs = X[ed]
                if not pred(Xs[1]): continue
                Ss = np.zeros((3, 3))
                for s, w in zip(fs.GL_X, fs.GL_W):
                    dM1 = np.array([4 * s - 3, 4 - 8 * s, 4 * s - 1]); dx = dM1 @ Xs; js = np.linalg.norm(dx)
                    Ss += w * js * np.outer(dM1 / js, dM1 / js); Bsz[ed] += w * js * (dM1 / js) * (dx / js)[1]
                r.append(np.repeat(ed, 3)); c.append(np.tile(ed, 3)); v.append(Ss.ravel())
    return sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(nd, nd)), Bsz
from scipy.sparse.linalg import spsolve
def solveW(A, Bss, Bsz, Lam, L, Tt, d):
    a = 1 + 3 * Lam * L**2; b = Lam * (1 + Tt)
    T = spsolve((a * A['K'] + b * A['M'] - a * d * Bss).tocsc().astype(complex),
                (-b * A['Fz'] - 3 * Lam * L**2 * A['Bnz'] + 3 * Lam * L**2 * d * Bsz).astype(complex))
    return (A['Fz'] @ T) / A['Fzz']
import json
true = {(r['B'], r['L'], round(r['Om'], 3)): r['true'] for r in json.load(open('../rect_omega.json'))}
for B in (1, 4):
    A = fs.assemble(fm.rect_mesh(B, 1.0, 24 * B, 24))
    side = split_boundary(A, lambda x: abs(abs(x[0]) - B / 2) < 1e-9)
    for L in (0.05, 0.1):
        for Om in (0.316, 316.228):
            Lam = 1j * Om; Tt = 1j * Om * 1e-3 * L**2; I1 = np.imag(g.slab_I(Lam, L, Tt, 'consistent'))
            c = lambda I: B * (np.imag(I) - I1) / I1
            print(f"  b/h={B} l/h={L} Om={Om:7.2f}: mixed FE {B*true[(B, L, Om)]:+.4f} | W all walls {c(fs.solve(A, Lam, L, Tt, 'W', 0.0)[0]):+.4f} | "
                  f"W side walls only {c(solveW(A, side[0], side[1], Lam, L, Tt, L)):+.4f} | eff. formula {c(g.rect_I_eff(Lam, L, Tt, B, 0.0)):+.4f} | linearised {c(g.rect_I_wall(Lam, L, Tt, B, 0.0)):+.4f}")

print("== 7. wall model on rectangles: dependence on delta at Om=0.316 (looking for resonances of the spurious boundary mode)")
for B in (1, 2, 4):
    A = fs.assemble(fm.rect_mesh(B, 1.0, 24 * B, 24)); row = []
    for L in np.arange(0.04, 0.1601, 0.01):
        Lam = 0.316j; Tt = 0.0; I1 = np.imag(g.slab_I(Lam, L, Tt, 'consistent'))
        row.append(B * (np.imag(fs.solve(A, Lam, L, Tt, 'W', 0.0)[0]) - I1) / I1 / (2 * L))
    print(f"  b/h={B}: (b/h) dQ/Q / (2 delta) for delta=0.04..0.16: " + " ".join(f"{x:+.2f}" for x in row))
