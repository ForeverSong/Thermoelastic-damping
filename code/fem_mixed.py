"""Mixed finite element method for consistent G-K TED (non-dimensional, source z):
    (1+Tt)(q,w) + L^2 a_DG(q,w) + bc(q,w) - (T, div w) = 0          for all w in BDM1
    Lam (T, s) + (div q, s) = -Lam (z, s)                             for all s in P0
a_DG = sum_K (2 eps(q):eps(w) + div q div w) + SIPG on interior edges (tangential jumps).
Boundary: q.n = 0 imposed strongly on (straight) boundary edges; tangential:
    S = inf  perfect slip (natural),  0 < S < inf  Robin (L^2/S) <q_t, w_t>,
    S = 0    no-slip by Nitsche (symmetric, penalty eta/h).
Discrete energy identity: Im (z, T_h) = -D_h / Omega with D_h = Re[(q,q) + L^2 a_DG(q,q) + bc(q,q)] >= 0.
BDM1 basis is built element-wise in physical space from scaled monomials (exact for affine elements).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import fem_mesh as fm

Q3P = np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]])      # degree-2 rule (reference)
Q3W = np.array([1/6, 1/6, 1/6])
G2 = np.array([0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3)])
G2W = np.array([0.5, 0.5])


def _monos(x, xc, hk):
    """6 vector monomials at points x (n,2): returns (n, 6, 2)."""
    xi = (x[:, 0] - xc[0]) / hk
    et = (x[:, 1] - xc[1]) / hk
    one, zero = np.ones_like(xi), np.zeros_like(xi)
    m = np.stack([np.stack([one, zero], -1), np.stack([xi, zero], -1), np.stack([et, zero], -1),
                  np.stack([zero, one], -1), np.stack([zero, xi], -1), np.stack([zero, et], -1)], axis=1)
    return m


def _mono_grads(hk):
    """gradients (6, 2, 2): G[j, a, b] = d m_j[a] / d x_b"""
    G = np.zeros((6, 2, 2))
    G[1, 0, 0] = 1 / hk; G[2, 0, 1] = 1 / hk; G[4, 1, 0] = 1 / hk; G[5, 1, 1] = 1 / hk
    return G


def build(mesh):
    p, t = mesh['p'], mesh['t']
    e, t2e, bnd = fm.edges(t)
    ne, nt = len(e), len(t)
    # global edge geometry: a<b orientation
    A, B = p[e[:, 0]], p[e[:, 1]]
    d = B - A
    elen = np.linalg.norm(d, axis=1)
    nE = np.c_[d[:, 1], -d[:, 0]] / elen[:, None]
    loc = np.array([[1, 2], [2, 0], [0, 1]])
    els = []
    for k in range(nt):
        X = p[t[k]]
        xc = X.mean(0)
        area = 0.5 * ((X[1, 0] - X[0, 0]) * (X[2, 1] - X[0, 1]) - (X[1, 1] - X[0, 1]) * (X[2, 0] - X[0, 0]))
        hk = np.sqrt(2 * area)
        Dm = np.zeros((6, 6))
        gdofs = np.zeros(6, int)
        for j in range(3):
            E = t2e[k, j]
            gdofs[2 * j], gdofs[2 * j + 1] = 2 * E, 2 * E + 1
            a, b = p[e[E, 0]], p[e[E, 1]]
            xs = a[None, :] + G2[:, None] * (b - a)[None, :]
            mv = _monos(xs, xc, hk)                    # (2,6,2)
            fl = mv @ nE[E]                             # (2,6) normal component
            Dm[2 * j] = elen[E] * (G2W @ fl)
            Dm[2 * j + 1] = elen[E] * ((G2W * (2 * G2 - 1)) @ fl)
        C = np.linalg.inv(Dm)                          # basis phi_i = sum_j C[j,i] m_j
        Gm = _mono_grads(hk)
        grad = np.einsum('ji,jab->iab', C, Gm)         # (6,2,2) constant gradients
        div = np.trace(grad, axis1=1, axis2=2)
        eps = 0.5 * (grad + grad.transpose(0, 2, 1))
        J = np.array([X[1] - X[0], X[2] - X[0]]).T
        xq = X[0][None, :] + Q3P @ J.T
        mq = _monos(xq, xc, hk)                        # (3,6,2)
        phq = np.einsum('ji,qjc->qic', C, mq)          # (3,6,2)
        Mq = 2 * area * np.einsum('q,qic,qjc->ij', Q3W, phq, phq)
        zint = area * xc[1]                            # int_K z (exact for linear z)
        zz = 2 * area * np.sum(Q3W * xq[:, 1] ** 2)
        els.append(dict(X=X, xc=xc, hk=hk, area=area, C=C, grad=grad, div=div, eps=eps, M=Mq,
                        gd=gdofs, zint=zint, zz=zz))
    # edge -> adjacent elements (with local edge index)
    adj = [[] for _ in range(ne)]
    for k in range(nt):
        for j in range(3):
            adj[t2e[k, j]].append((k, j))
    return dict(p=p, t=t, e=e, t2e=t2e, bnd=bnd, nE=nE, elen=elen, els=els, adj=adj, ne=ne, nt=nt)


def _trace(el, x):
    """basis values (n,6,2) of element el at points x"""
    return np.einsum('ji,qjc->qic', el['C'], _monos(x, el['xc'], el['hk']))


def assemble(S, L, eta=10.0, Tt=0.0, marker=None):
    """Assemble the parts that do not depend on Lam. Returns blocks for later solves."""
    ne, nt = S['ne'], S['nt']
    nq = 2 * ne
    R, Cc, Vm, Vv, Vb = [], [], [], [], []
    Bdiv = sp.lil_matrix((nt, nq))
    z = np.zeros(nt); zz = 0.0; area = np.zeros(nt)
    for k, el in enumerate(S['els']):
        gd = el['gd']
        V = el['area'] * (2 * np.einsum('iab,jab->ij', el['eps'], el['eps']) + np.outer(el['div'], el['div']))
        R.append(np.repeat(gd, 6)); Cc.append(np.tile(gd, 6))
        Vm.append(el['M'].ravel()); Vv.append(V.ravel()); Vb.append((el['area'] * np.outer(el['div'], el['div'])).ravel())
        Bdiv[k, gd] = el['area'] * el['div']
        z[k] = el['zint']; zz += el['zz']; area[k] = el['area']
    R, Cc = np.concatenate(R), np.concatenate(Cc)
    Mq = sp.csr_matrix((np.concatenate(Vm), (R, Cc)), shape=(nq, nq))
    Vq = sp.csr_matrix((np.concatenate(Vv), (R, Cc)), shape=(nq, nq))
    Vd = sp.csr_matrix((np.concatenate(Vb), (R, Cc)), shape=(nq, nq))      # (div, div): weight (alpha - 1)
    # interior-edge SIPG and boundary terms
    Jr, Jc, Jv = [], [], []
    Rr, Rc, Rv = [], [], []               # boundary tangential mass <q_t, w_t>
    Kr, Kc, Kv = [], [], []               # curvature-weighted tangential mass <kappa q_t, w_t>
    Nr, Nc, Nv = [], [], []               # boundary Nitsche (no-slip) terms without penalty scaling
    for E in range(ne):
        a, b = S['p'][S['e'][E, 0]], S['p'][S['e'][E, 1]]
        xs = a[None, :] + G2[:, None] * (b - a)[None, :]
        lenE = S['elen'][E]
        nF = S['nE'][E]
        if not S['bnd'][E]:
            (k1, _), (k2, _) = S['adj'][E]
            e1, e2 = S['els'][k1], S['els'][k2]
            hF = 0.5 * (e1['area'] + e2['area']) / lenE
            if np.dot(nF, 0.5 * (a + b) - e1['xc']) < 0:                          # n_F outward of K1 (= e1)
                nF = -nF
            tr = np.concatenate([_trace(e1, xs), -_trace(e2, xs)], axis=1)       # jump contributions (2,12,2)
            av = 0.5 * np.concatenate([np.einsum('iab,b->ia', 2 * e1['eps'], nF),
                                       np.einsum('iab,b->ia', 2 * e2['eps'], nF)], axis=0)  # (12,2)
            jj = lenE * np.einsum('q,qic,qjc->ij', G2W, tr, tr)
            aj = lenE * np.einsum('q,jc,qic->ij', G2W, av, tr)                   # int A_j . J_i
            Mloc = -aj - aj.T + (eta / hF) * jj
            gd = np.concatenate([e1['gd'], e2['gd']])
            Jr.append(np.repeat(gd, 12)); Jc.append(np.tile(gd, 12)); Jv.append(Mloc.ravel())
        else:
            (k1, _), = S['adj'][E]
            e1 = S['els'][k1]
            n_out = nF if np.dot(nF, (0.5 * (a + b) - e1['xc'])) > 0 else -nF
            tv = np.array([-n_out[1], n_out[0]])
            hF = e1['area'] / lenE
            tr = _trace(e1, xs)                                                   # (2,6,2)
            tt = tr @ tv                                                          # (2,6)
            Mt = lenE * np.einsum('q,qi,qj->ij', G2W, tt, tt)
            tra = np.einsum('iab,b->ia', 2 * e1['eps'], n_out) @ tv               # (6,)
            Ant = lenE * np.einsum('q,j,qi->ij', G2W, tra, tt)                    # int (2eps(phi_j)n).t * phi_i.t
            gd = e1['gd']
            kind = 'wall' if marker is None else marker(0.5 * (a + b))
            kap = S.get('kappa', 0.0)
            Kr.append(np.repeat(gd, 6)); Kc.append(np.tile(gd, 6)); Kv.append((kap * Mt).ravel())
            if kind == 'wall':
                Rr.append(np.repeat(gd, 6)); Rc.append(np.tile(gd, 6)); Rv.append(Mt.ravel())
                Nloc = -Ant - Ant.T + (eta / hF) * Mt
                Nr.append(np.repeat(gd, 6)); Nc.append(np.tile(gd, 6)); Nv.append(Nloc.ravel())
            # kind == 'free': zero tangential traction (natural), nothing added
    Jq = sp.csr_matrix((np.concatenate(Jv), (np.concatenate(Jr), np.concatenate(Jc))), shape=(nq, nq))
    Bt = sp.csr_matrix((np.concatenate(Rv), (np.concatenate(Rr), np.concatenate(Rc))), shape=(nq, nq)) if Rv else sp.csr_matrix((nq, nq))
    Bn = sp.csr_matrix((np.concatenate(Nv), (np.concatenate(Nr), np.concatenate(Nc))), shape=(nq, nq)) if Nv else sp.csr_matrix((nq, nq))
    Kt = sp.csr_matrix((np.concatenate(Kv), (np.concatenate(Kr), np.concatenate(Kc))), shape=(nq, nq)) if Kv else sp.csr_matrix((nq, nq))
    free = np.ones(nq, bool)
    bE = np.where(S['bnd'])[0]
    free[2 * bE] = False; free[2 * bE + 1] = False
    return dict(Mq=Mq, Vq=Vq, Vd=Vd, Jq=Jq, Bt=Bt, Bn=Bn, Kt=Kt, B=Bdiv.tocsr(), z=z, zz=zz, area=area, free=free, nq=nq, nt=nt)


def solve(Asm, Lam, L, Tt, Sslip, curvature_fix=False, alpha=2.0):
    """Sslip = ls/a: 0 -> no-slip (Nitsche), inf -> perfect slip, else Robin."""
    Aqq = (1 + Tt) * Asm['Mq'] + L ** 2 * (Asm['Vq'] + (alpha - 2.0) * Asm['Vd'] + Asm['Jq'])   # 2(eps,eps) + (alpha-1)(div,div)
    if Sslip == 0:
        Aqq = Aqq + L ** 2 * Asm['Bn']
    elif not np.isinf(Sslip):
        Aqq = Aqq + (L ** 2 / Sslip) * Asm['Bt']
    if curvature_fix and Sslip != 0:
        # polygonal walls carry no curvature: restore the -kappa q_t term of the curved-wall traction
        Aqq = Aqq - L ** 2 * Asm['Kt']
    f = Asm['free']
    Aff = Aqq[f][:, f]
    Bf = Asm['B'][:, f]
    nt = Asm['nt']
    K = sp.bmat([[Aff, -Bf.T], [Bf, Lam * sp.diags(Asm['area'])]], format='csc')
    rhs = np.concatenate([np.zeros(f.sum(), complex), -Lam * Asm['z']])
    x = spsolve(K.astype(complex), rhs)
    qf, T = x[:f.sum()], x[f.sum():]
    q = np.zeros(Asm['nq'], complex); q[f] = qf
    I = (Asm['z'] @ T) / Asm['zz']
    D = np.real(np.conj(q) @ (Aqq @ q))
    return dict(I=I, T=T, q=q, D=D, Omega=np.imag(Lam))
