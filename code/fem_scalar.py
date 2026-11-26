"""Isoparametric P2 Lagrange finite elements for the scalar (temperature-only) G-K TED problem
with three boundary closures (non-dimensional, source theta = z):
    'S'  literature:  dT/dn = 0
    'P'  consistent normal flux, vorticity neglected:  (1+3 Lam L^2) dT/dn = -3 Lam L^2 n_z
    'W'  Knudsen-layer wall model (Wentzell):
         (1+3 Lam L^2)(dT/dn + delta d2T/ds2) = -3 Lam L^2 (n_z + delta d2z/ds2)
Bulk: (1+3 Lam L^2) lap T - Lam (1+Tt) T = Lam (1+Tt) z.
Weak form (closed or piecewise-smooth boundary, edge-wise tangential integration by parts):
    a (grad T, grad v) + b (T, v) - a delta <dT/ds, dv/ds> = -b (z, v) - 3 Lam L^2 <n_z, v> + 3 Lam L^2 delta <t_z, dv/ds>
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import fem_mesh as fm

# 7-point degree-5 rule on the reference triangle
_a, _b = 0.059715871789770, 0.470142064105115
_c, _d = 0.797426985353087, 0.101286507323456
QP = np.array([[1/3, 1/3], [_b, _b], [_a, _b], [_b, _a], [_d, _d], [_c, _d], [_d, _c]])
QW = 0.5 * np.array([0.225, 0.132394152788506, 0.132394152788506, 0.132394152788506,
                     0.125939180544827, 0.125939180544827, 0.125939180544827])
GL_X, GL_W = np.polynomial.legendre.leggauss(5)
GL_X, GL_W = 0.5 * (GL_X + 1), 0.5 * GL_W


def p2_basis(xi, eta):
    l0, l1, l2 = 1 - xi - eta, xi, eta
    N = np.array([l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), 4 * l1 * l2, 4 * l2 * l0, 4 * l0 * l1])
    dl = np.array([[-1, -1], [1, 0], [0, 1]], float)
    dN = np.array([(4 * l0 - 1) * dl[0], (4 * l1 - 1) * dl[1], (4 * l2 - 1) * dl[2],
                   4 * (l1 * dl[2] + l2 * dl[1]), 4 * (l2 * dl[0] + l0 * dl[2]), 4 * (l0 * dl[1] + l1 * dl[0])])
    return N, dN


def p2_mesh(mesh):
    p, t = mesh['p'], mesh['t']
    e, t2e, bnd = fm.edges(t)
    N = len(p)
    mid = 0.5 * (p[e[:, 0]] + p[e[:, 1]])
    if mesh.get('curved'):
        R = mesh['R']
        mid[bnd] = mid[bnd] / np.linalg.norm(mid[bnd], axis=1)[:, None] * R
    X = np.vstack([p, mid]) * np.asarray(mesh.get('scale', (1.0, 1.0)), float)
    dofs = np.hstack([t, N + t2e])               # local order [v0,v1,v2,m(v1v2),m(v2v0),m(v0v1)]
    return dict(X=X, dofs=dofs, e=e, t2e=t2e, bnd=bnd, nv=N, t=t)


def assemble(mesh):
    m = p2_mesh(mesh)
    X, dofs = m['X'], m['dofs']
    nd = len(X)
    rows, cols, Kv, Mv = [], [], [], []
    Fz = np.zeros(nd); Fzz = 0.0; area = 0.0
    B = [p2_basis(*q) for q in QP]
    for k, el in enumerate(dofs):
        Xe = X[el]
        Ke = np.zeros((6, 6)); Me = np.zeros((6, 6))
        for (N, dN), w in zip(B, QW):
            J = Xe.T @ dN                         # 2x2
            dJ = np.linalg.det(J)
            G = dN @ np.linalg.inv(J)             # 6x2 physical gradients
            x = N @ Xe
            Ke += w * dJ * G @ G.T
            Me += w * dJ * np.outer(N, N)
            Fz[el] += w * dJ * N * x[1]
            Fzz += w * dJ * x[1] ** 2
            area += w * dJ
        rows.append(np.repeat(el, 6)); cols.append(np.tile(el, 6))
        Kv.append(Ke.ravel()); Mv.append(Me.ravel())
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    K = sp.csr_matrix((np.concatenate(Kv), (rows, cols)), shape=(nd, nd))
    M = sp.csr_matrix((np.concatenate(Mv), (rows, cols)), shape=(nd, nd))
    # boundary edges, oriented counter-clockwise (element's local orientation)
    loc = [(1, 2, 3), (2, 0, 4), (0, 1, 5)]
    br, bc, Bss, Mbv, Cv = [], [], [], [], []
    Bnz = np.zeros(nd); Bsz = np.zeros(nd); Btz = np.zeros(nd); perim = 0.0
    for k in range(len(dofs)):
        for j in range(3):
            if m['bnd'][m['t2e'][k, j]]:
                a, b_, c = loc[j]
                ed = dofs[k][[a, c, b_]]          # start, mid, end
                Xs = X[ed]
                Ss = np.zeros((3, 3)); Ms = np.zeros((3, 3)); Cs_ = np.zeros((3, 3))
                for s, w in zip(GL_X, GL_W):
                    M1 = np.array([(1 - s) * (1 - 2 * s), 4 * s * (1 - s), s * (2 * s - 1)])
                    dM1 = np.array([4 * s - 3, 4 - 8 * s, 4 * s - 1])
                    dx = dM1 @ Xs
                    js = np.linalg.norm(dx)
                    tvec = dx / js
                    nvec = np.array([tvec[1], -tvec[0]])
                    ds_v = dM1 / js
                    Ss += w * js * np.outer(ds_v, ds_v)
                    Ms += w * js * np.outer(M1, M1)
                    Cs_ += w * js * np.outer(M1, ds_v)
                    Btz[ed] += w * js * M1 * tvec[1]
                    Bnz[ed] += w * js * M1 * nvec[1]
                    Bsz[ed] += w * js * ds_v * tvec[1]
                    perim += w * js
                br.append(np.repeat(ed, 3)); bc.append(np.tile(ed, 3)); Bss.append(Ss.ravel()); Mbv.append(Ms.ravel()); Cv.append(Cs_.ravel())
    Bss = sp.csr_matrix((np.concatenate(Bss), (np.concatenate(br), np.concatenate(bc))), shape=(nd, nd))
    Mb = sp.csr_matrix((np.concatenate(Mbv), (np.concatenate(br), np.concatenate(bc))), shape=(nd, nd))
    Cmat = sp.csr_matrix((np.concatenate(Cv), (np.concatenate(br), np.concatenate(bc))), shape=(nd, nd))
    bidx = np.unique(np.concatenate(br))
    return dict(K=K, M=M, Fz=Fz, Fzz=Fzz, Bss=Bss, Mb=Mb, Cmat=Cmat, Btz=Btz, bidx=bidx, Bnz=Bnz, Bsz=Bsz, area=area, perim=perim, mesh=m)


def solve(A, Lam, L, Tt, model, Cs=0.0, alpha=2.0):
    a = 1 + (1 + alpha) * Lam * L ** 2
    b = Lam * (1 + Tt)
    if model == 'S':
        Mat = a * A['K'] + b * A['M']
        rhs = -b * A['Fz']
    else:
        if model == 'P':
            d = 0.0
        elif model == 'W':
            beta = np.sqrt(1 + Tt)
            d = 0.0 if np.isinf(Cs) else L / (beta * (1 + Cs * beta))
        else:
            raise ValueError(model)
        Mat = a * A['K'] + b * A['M'] - a * d * A['Bss']
        rhs = -b * A['Fz'] - (1 + alpha) * Lam * L ** 2 * A['Bnz'] + (1 + alpha) * Lam * L ** 2 * d * A['Bsz']
    T = spsolve(Mat.tocsc().astype(complex), rhs.astype(complex))
    I = (A['Fz'] @ T) / A['Fzz']
    return I, T
