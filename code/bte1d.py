"""Kinetic reference for the slab: gray single-relaxation-time phonon Boltzmann equation, frequency domain.

Non-dimensional (lengths by h, time by h^2/chi, chi = v*mfp/3, Kn = mfp/h):
    (1 + tt) g + Kn mu dg/dz = T - tt z,    T = (1/2) int g dmu,    tt = i Omega Kn^2 / 3
The isotropic source -tt z is the thermoelastic (Grueneisen) coupling; its zeroth moment gives
    i Omega (T + z) + dq/dz = 0,    q = (3/Kn)(1/2) int mu g dmu   (so q = -dT/dz in the diffusive limit).
Walls z = +-1/2 are adiabatic: specular (g(mu) = g(-mu)) or diffuse (re-emission isotropic, zero net flux).
Matching the second-order moment equations gives the equivalent G-K parameters in one dimension:
    tau = mfp/v  ->  tt as above,     3 l^2 = (4/15) mfp^2  ->  L^2 = 4 Kn^2 / 45.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import gkted as g


def gk_params(Om, Kn):
    return 1j * Om, np.sqrt(4.0 / 45.0) * Kn, 1j * Om * Kn ** 2 / 3.0


def bte_specular(Om, Kn, N=100000):
    """Specular walls = even periodic extension: each Fourier mode k_n = (2n+1) pi is solved in closed form,
    T_n/z_n = -tt A/(1 - A),  A = arctan(x/(1+tt))/x,  x = Kn k_n;   I = 96 sum_n (T_n/z_n)/k_n^4."""
    tt = 1j * Om * Kn ** 2 / 3.0
    k = (2 * np.arange(N) + 1) * np.pi
    x = Kn * k
    A = np.arctan(x / (1 + tt)) / x
    return 96.0 * np.sum(-tt * A / (1 - A) / k ** 4)


def grid(N=400, s=2.5):
    xi = np.linspace(-1, 1, N + 1)
    return 0.5 * np.tanh(s * xi) / np.tanh(s)


def bte_dom(Om, Kn, wall='diffuse', N=400, Nmu=24, s=2.5, return_fields=False):
    """Discrete ordinates (Gauss-Legendre on each half range) + step-characteristic scheme in z (conservative)."""
    tt = 1j * Om * Kn ** 2 / 3.0
    xe = grid(N, s); dz = np.diff(xe); zc = 0.5 * (xe[1:] + xe[:-1])
    x, w = np.polynomial.legendre.leggauss(Nmu); mu_h = 0.5 * (x + 1); w_h = 0.5 * w        # (0,1), weights sum to 1
    mu = np.concatenate([mu_h, -mu_h]); wt = np.concatenate([w_h, w_h]); M = 2 * Nmu
    W1 = np.sum(w_h * mu_h)
    nG = M * (N + 1); iT = nG + np.arange(N); n = nG + N
    gi = lambda j, e: j * (N + 1) + e
    R, C, V = [], [], []; rhs = np.zeros(n, complex)
    add = lambda r, c, v: (R.append(np.asarray(r).ravel()), C.append(np.asarray(c).ravel()), V.append(np.asarray(v, complex).ravel()))
    cells = np.arange(N); Tdiag = np.ones(N, complex); row = 0
    for j in range(M):
        eps = (1 + tt) * dz / (Kn * abs(mu[j])); E = np.exp(-eps); Cc = (1 - E) / (1 + tt); P = -np.expm1(-eps) / eps
        e_in, e_out = (cells, cells + 1) if mu[j] > 0 else (cells + 1, cells)
        rows = row + cells
        add(rows, gi(j, e_out), np.ones(N)); add(rows, gi(j, e_in), -E); add(rows, iT, -Cc)
        rhs[rows] = -Cc * tt * zc; row += N
        # temperature equation contributions
        add(iT, gi(j, e_in), -0.5 * wt[j] * P)
        Tdiag -= 0.5 * wt[j] * (1 - P) / (1 + tt)
        rhs[iT] += -0.5 * wt[j] * (1 - P) * tt * zc / (1 + tt)
        # wall condition for the incoming direction
        e_w = 0 if mu[j] > 0 else N
        add([row], [gi(j, e_w)], [1.0])
        if wall == 'specular':
            jm = j + Nmu if j < Nmu else j - Nmu
            add([row], [gi(jm, e_w)], [-1.0])
        else:
            out = np.arange(Nmu, M) if mu[j] > 0 else np.arange(Nmu)
            add(np.full(Nmu, row), gi(out, e_w), -wt[out] * np.abs(mu[out]) / W1)
        row += 1
    add(iT, iT, Tdiag)
    Amat = sp.csc_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(n, n))
    sol = spsolve(Amat, rhs)
    T = sol[iT]
    I = 12.0 * np.sum(T * zc * dz)
    if return_fields:
        G = sol[:nG].reshape(M, N + 1)
        q = (3.0 / Kn) * 0.5 * (wt * mu) @ G
        return dict(I=I, zc=zc, T=T, xe=xe, q=q, G=G, mu=mu, w=wt)
    return I


def closures(Om, Kn):
    Lam, L, tt = gk_params(Om, Kn)
    return dict(fourier=g.slab_I(Lam, 0.0, 0.0, 'S'), cv=g.slab_I(Lam, 0.0, tt, 'S'),
                scalar=g.slab_I(Lam, L, tt, 'S'), consistent=g.slab_I(Lam, L, tt, 'consistent'))


if __name__ == '__main__':
    import time
    print("solver check: discrete ordinates with specular walls vs closed-form mode sum")
    for Om, Kn in ((1.0, 0.05), (10.0, 0.1), (10.0, 0.5), (100.0, 0.2), (3.0, 1.0)):
        ex = bte_specular(Om, Kn); t0 = time.time()
        a = bte_dom(Om, Kn, 'specular', N=200, Nmu=16); b = bte_dom(Om, Kn, 'specular', N=400, Nmu=24)
        print(f"  Om={Om:6.1f} Kn={Kn:4.2f}: I={ex:.8f}  rel.err N=200,Nmu=16: {abs(a-ex)/abs(ex):.1e}   N=400,Nmu=24: {abs(b-ex)/abs(ex):.1e}  ({time.time()-t0:.1f}s)")
    print("wall fields, Om=10, Kn=0.1 (specular): flux and temperature gradient at the wall")
    F = bte_dom(10.0, 0.1, 'specular', return_fields=True); I, zc, T, xe, q = F['I'], F['zc'], F['T'], F['xe'], F['q']
    Lam, L, tt = gk_params(10.0, 0.1)
    print(f"  BTE: |q(wall)|={abs(q[-1]):.1e}   dT/dz at wall (last two cells) = {(T[-1]-T[-2])/(zc[-1]-zc[-2]):.4f}   at z=0.45: {np.interp(0.45, zc, np.gradient(T, zc).real):.4f}{np.interp(0.45, zc, np.gradient(T, zc).imag):+.4f}j")
    print(f"  consistent G-K wall gradient -3 Lam L^2/(1+3 Lam L^2) = {-3*Lam*L**2/(1+3*Lam*L**2):.4f}")
