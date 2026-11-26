"""Mixed BDM1-P0 method on P2-isoparametric (curved) triangles with the contravariant Piola map.
Same weak form as fem_mixed.py, but q.n = 0 is imposed exactly on the curved (P2) boundary,
avoiding the Babuska-type geometric error of polygonal boundaries.
    phi(x) = J(xh) phih(xh) / det J(xh)    (normal moments preserved on every edge)
Energy identity Im(z, T_h) = -D_h/Omega holds exactly (it only needs the integral energy balance).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import fem_mesh as fm
from fem_scalar import QP, QW, p2_basis

GX, GW = np.polynomial.legendre.leggauss(4)
GX, GW = 0.5 * (GX + 1), 0.5 * GW

# P2 second derivatives (constant): H[n] = [[d_xx, d_xy], [d_xy, d_yy]]
H2 = np.array([[[4, 4], [4, 4]], [[4, 0], [0, 0]], [[0, 0], [0, 4]], [[0, 4], [4, 0]],
               [[0, -4], [-4, -8]], [[-8, -4], [-4, 0]]], float)

# reference BDM1 basis
REF_V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
LOC_EDGES = [(1, 2), (2, 0), (0, 1)]          # local edge j opposite vertex j, CCW


def _mono(x):
    xi, et = x[..., 0], x[..., 1]
    one, zero = np.ones_like(xi), np.zeros_like(xi)
    return np.stack([np.stack([one, zero], -1), np.stack([xi, zero], -1), np.stack([et, zero], -1),
                     np.stack([zero, one], -1), np.stack([zero, xi], -1), np.stack([zero, et], -1)], axis=-2)


MONO_GRAD = np.zeros((6, 2, 2))
MONO_GRAD[1, 0, 0] = 1; MONO_GRAD[2, 0, 1] = 1; MONO_GRAD[4, 1, 0] = 1; MONO_GRAD[5, 1, 1] = 1


def _ref_basis():
    D = np.zeros((6, 6))
    for j, (a, b) in enumerate(LOC_EDGES):
        A, B = REF_V[a], REF_V[b]
        d = B - A
        n = np.array([d[1], -d[0]]) / np.linalg.norm(d)
        ln = np.linalg.norm(d)
        xs = A[None, :] + GX[:, None] * d[None, :]
        fl = _mono(xs) @ n                       # (q,6)
        D[2 * j] = ln * (GW @ fl)
        D[2 * j + 1] = ln * ((GW * (2 * GX - 1)) @ fl)
    return np.linalg.inv(D)


CREF = _ref_basis()                                # phih_i = sum_j CREF[j,i] m_j
GRADREF = np.einsum('ji,jab->iab', CREF, MONO_GRAD)  # (6,2,2) d phih_i[a]/d xh_b
DIVREF = np.trace(GRADREF, axis1=1, axis2=2)       # (6,)


def _phih(xh):
    return np.einsum('ji,...jc->...ic', CREF, _mono(xh))


def _geom(Xe, xh):
    """map, Jacobian, det, d_k J, d_k det at reference point xh."""
    N, dN = p2_basis(*xh)
    x = N @ Xe
    J = Xe.T @ dN                                  # J[a,b] = dx_a/dxh_b
    dJ = np.einsum('na,nbk->kab', Xe, H2)          # dJ[k,a,b] = d/dxh_k J[a,b]
    D = np.linalg.det(J)
    dD = dJ[:, 0, 0] * J[1, 1] + J[0, 0] * dJ[:, 1, 1] - dJ[:, 0, 1] * J[1, 0] - J[0, 1] * dJ[:, 1, 0]
    return x, J, D, dJ, dD


def _phys(Xe, xh, signs):
    """physical basis values (6,2), gradients (6,2,2) at reference point xh; signs (6,)"""
    x, J, D, dJ, dD = _geom(Xe, xh)
    ph = _phih(xh)                                 # (6,2)
    val = (ph @ J.T) / D                           # J phih / D
    dval = np.zeros((6, 2, 2))                     # d phi[a]/d xh_k
    for k in range(2):
        dval[:, :, k] = (ph @ dJ[k].T + GRADREF[:, :, k] @ J.T) / D - (ph @ J.T) * dD[k] / D ** 2
    Ji = np.linalg.inv(J)
    grad = np.einsum('iak,kb->iab', dval, Ji)
    return x, D, signs[:, None] * val, signs[:, None, None] * grad, J


def build(mesh):
    p, t = mesh['p'], mesh['t']
    e, t2e, bnd = fm.edges(t)
    ne, nt = len(e), len(t)
    mid = 0.5 * (p[e[:, 0]] + p[e[:, 1]])
    if mesh.get('curved'):
        mid[bnd] = mid[bnd] / np.linalg.norm(mid[bnd], axis=1)[:, None] * mesh['R']
    sc = np.asarray(mesh.get('scale', (1.0, 1.0)), float)      # affine image of the disk: elliptical sections
    p = p * sc; mid = mid * sc
    els = []
    for k in range(nt):
        Xe = np.vstack([p[t[k]], mid[t2e[k, 0]], mid[t2e[k, 1]], mid[t2e[k, 2]]])
        gd = np.zeros(6, int); sg = np.ones(6)
        for j, (a, b) in enumerate(LOC_EDGES):
            E = t2e[k, j]
            gd[2 * j], gd[2 * j + 1] = 2 * E, 2 * E + 1
            if t[k, a] > t[k, b]:                    # local orientation opposite to global (low -> high)
                sg[2 * j] = -1.0
        curved = bool(mesh.get('curved')) and bnd[t2e[k]].any()
        els.append(dict(Xe=Xe, gd=gd, sg=sg, curved=curved))
    adj = [[] for _ in range(ne)]
    for k in range(nt):
        for j in range(3):
            adj[t2e[k, j]].append((k, j))
    return dict(p=p, t=t, e=e, t2e=t2e, bnd=bnd, els=els, adj=adj, ne=ne, nt=nt)


def _edge_points(el, j, sigma, start_is_a):
    """reference points on local edge j for global parameter sigma (a -> b)."""
    a, b = LOC_EDGES[j]
    s = sigma if start_is_a else 1 - sigma
    return REF_V[a][None, :] + s[:, None] * (REF_V[b] - REF_V[a])[None, :], s


def assemble(S, L, eta=10.0, marker=None):
    ne, nt = S['ne'], S['nt']
    nq = 2 * ne
    R, Cc, Vm, Vv, Vdv = [], [], [], [], []
    Bd = sp.lil_matrix((nt, nq))
    z = np.zeros(nt); zz = 0.0; area = np.zeros(nt)
    for k, el in enumerate(S['els']):
        Me = np.zeros((6, 6)); Ve = np.zeros((6, 6)); Vde = np.zeros((6, 6))
        for xh, w in zip(QP, QW):
            x, D, val, grad, _ = _phys(el['Xe'], xh, el['sg'])
            eps = 0.5 * (grad + grad.transpose(0, 2, 1))
            div = np.trace(grad, axis1=1, axis2=2)
            Me += w * D * val @ val.T
            Ve += w * D * (2 * np.einsum('iab,jab->ij', eps, eps) + np.outer(div, div))
            Vde += w * D * np.outer(div, div)
            z[k] += w * D * x[1]; zz += w * D * x[1] ** 2; area[k] += w * D
        gd = el['gd']
        R.append(np.repeat(gd, 6)); Cc.append(np.tile(gd, 6)); Vm.append(Me.ravel()); Vv.append(Ve.ravel()); Vdv.append(Vde.ravel())
        Bd[k, gd] = el['sg'] * DIVREF * 0.5            # int_K div phi = divh(phih) * |ref|
    R, Cc = np.concatenate(R), np.concatenate(Cc)
    Mq = sp.csr_matrix((np.concatenate(Vm), (R, Cc)), shape=(nq, nq))
    Vq = sp.csr_matrix((np.concatenate(Vv), (R, Cc)), shape=(nq, nq))
    Vd = sp.csr_matrix((np.concatenate(Vdv), (R, Cc)), shape=(nq, nq))     # (div, div): weight (alpha - 1)
    Jr, Jc, Jv, Rr, Rc, Rv, Nr, Nc, Nv = ([] for _ in range(9))
    for E in range(ne):
        ga = S['e'][E, 0]
        if not S['bnd'][E]:
            (k1, j1), (k2, j2) = S['adj'][E]
            e1, e2 = S['els'][k1], S['els'][k2]
            xs1, s1 = _edge_points(e1, j1, GX, S['t'][k1, LOC_EDGES[j1][0]] == ga)
            xs2, s2 = _edge_points(e2, j2, GX, S['t'][k2, LOC_EDGES[j2][0]] == ga)
            v1, v2, t1, t2, dsl = [], [], [], [], []
            for q in range(len(GX)):
                x1, D1, val1, g1, J1 = _phys(e1['Xe'], xs1[q], e1['sg'])
                x2, D2, val2, g2, _ = _phys(e2['Xe'], xs2[q], e2['sg'])
                a_, b_ = LOC_EDGES[j1]
                tvec = J1 @ (REF_V[b_] - REF_V[a_])        # d x / d s (local, element 1)
                ds = np.linalg.norm(tvec)
                n1 = np.array([tvec[1], -tvec[0]]) / ds        # outward of element 1
                v1.append(val1); v2.append(val2)
                t1.append(np.einsum('iab,b->ia', 0.5 * (g1 + g1.transpose(0, 2, 1)) * 2, n1))
                t2.append(np.einsum('iab,b->ia', 0.5 * (g2 + g2.transpose(0, 2, 1)) * 2, n1))
                dsl.append(ds)
            v1, v2, t1, t2, dsl = map(np.array, (v1, v2, t1, t2, dsl))
            jump = np.concatenate([v1, -v2], axis=1)          # (q,12,2)
            avg = 0.5 * np.concatenate([t1, t2], axis=1)      # (q,12,2)
            wq = GW * dsl
            lenE = wq.sum()
            hF = 0.5 * (abs(np.linalg.det(np.c_[e1['Xe'][1] - e1['Xe'][0], e1['Xe'][2] - e1['Xe'][0]])) +
                        abs(np.linalg.det(np.c_[e2['Xe'][1] - e2['Xe'][0], e2['Xe'][2] - e2['Xe'][0]]))) / 2 / lenE
            jj = np.einsum('q,qic,qjc->ij', wq, jump, jump)
            aj = np.einsum('q,qjc,qic->ij', wq, avg, jump)
            Mloc = -aj - aj.T + (eta / hF) * jj
            gd = np.concatenate([e1['gd'], e2['gd']])
            Jr.append(np.repeat(gd, 12)); Jc.append(np.tile(gd, 12)); Jv.append(Mloc.ravel())
        else:
            (k1, j1), = S['adj'][E]
            e1 = S['els'][k1]
            xs1, _ = _edge_points(e1, j1, GX, True)          # local orientation (CCW)
            Mt = np.zeros((6, 6)); Ant = np.zeros((6, 6)); lenE = 0.0
            mids = []
            for q in range(len(GX)):
                x1, D1, val1, g1, J1 = _phys(e1['Xe'], xs1[q], e1['sg'])
                a_, b_ = LOC_EDGES[j1]
                tvec = J1 @ (REF_V[b_] - REF_V[a_])
                ds = np.linalg.norm(tvec)
                tv = tvec / ds
                nv = np.array([tv[1], -tv[0]])
                vt = val1 @ tv
                tra = np.einsum('iab,b->ia', g1 + g1.transpose(0, 2, 1), nv) @ tv
                Mt += GW[q] * ds * np.outer(vt, vt)
                Ant += GW[q] * ds * np.outer(vt, tra)          # [i,j] = int (2eps(phi_j)n).t phi_i.t
                lenE += GW[q] * ds
                mids.append(x1)
            hF = 0.5 * abs(np.linalg.det(np.c_[e1['Xe'][1] - e1['Xe'][0], e1['Xe'][2] - e1['Xe'][0]])) / lenE
            gd = e1['gd']
            kind = 'wall' if marker is None else marker(np.mean(mids, axis=0))
            if kind == 'wall':
                Rr.append(np.repeat(gd, 6)); Rc.append(np.tile(gd, 6)); Rv.append(Mt.ravel())
                Nloc = -Ant - Ant.T + (eta / hF) * Mt
                Nr.append(np.repeat(gd, 6)); Nc.append(np.tile(gd, 6)); Nv.append(Nloc.ravel())
    mk = lambda r, c, v: sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(nq, nq)) if v else sp.csr_matrix((nq, nq))
    free = np.ones(nq, bool)
    bE = np.where(S['bnd'])[0]
    free[2 * bE] = False; free[2 * bE + 1] = False
    return dict(Mq=Mq, Vq=Vq, Vd=Vd, Jq=mk(Jr, Jc, Jv), Bt=mk(Rr, Rc, Rv), Bn=mk(Nr, Nc, Nv), B=Bd.tocsr(),
                z=z, zz=zz, area=area, free=free, nq=nq, nt=nt)


def solve(Asm, Lam, L, Tt, Sslip, alpha=2.0):
    Aqq = (1 + Tt) * Asm['Mq'] + L ** 2 * (Asm['Vq'] + (alpha - 2.0) * Asm['Vd'] + Asm['Jq'])   # 2(eps,eps) + (alpha-1)(div,div)
    if Sslip == 0:
        Aqq = Aqq + L ** 2 * Asm['Bn']
    elif not np.isinf(Sslip):
        Aqq = Aqq + (L ** 2 / Sslip) * Asm['Bt']
    f = Asm['free']
    Aff = Aqq[f][:, f]
    Bf = Asm['B'][:, f]
    K = sp.bmat([[Aff, -Bf.T], [Bf, Lam * sp.diags(Asm['area'])]], format='csc')
    rhs = np.concatenate([np.zeros(f.sum(), complex), -Lam * Asm['z']])
    x = spsolve(K.astype(complex), rhs)
    q = np.zeros(Asm['nq'], complex); q[f] = x[:f.sum()]
    T = x[f.sum():]
    I = (Asm['z'] @ T) / Asm['zz']
    D = np.real(np.conj(q) @ (Aqq @ q))
    return dict(I=I, T=T, q=q, D=D, Omega=np.imag(Lam))
