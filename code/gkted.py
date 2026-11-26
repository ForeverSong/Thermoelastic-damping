"""
gkted.py -- Thermoelastic damping (TED) with Guyer-Krumhansl (G-K) heat conduction:
closed-form solutions with *consistent* heat-flux boundary conditions, and
independent spectral-collocation checks.

Problem (frequency domain, lambda = i*omega), per unit thermoelastic source s z:
    energy :  lambda*C*(T + z) + div q = 0
    G-K    :  (1 + tau*lambda) q - l^2 (lap q + 2 grad div q) + k grad T = 0
Boundary conditions considered
    'S'     scalar (temperature-only) formulation:  dT/dn = 0, q irrotational
    'P'     consistent normal condition q.n = 0, q irrotational (vorticity neglected)
    'slip'  consistent:  q.n = 0  and  q_t = -ls * t.(2 eps(q) n)     (ls = slip length)
            ls = 0 -> no-slip, ls = inf -> zero tangential traction ("perfect slip")
Inverse quality factor:  Q^-1 = Delta_E * |Im I|,  I = int z T dA / int z^2 dA.

Everything is non-dimensional. For the slab (rectangular beam, thickness h) lengths
are scaled by h; for the disk (circular beam, radius a) by a. Parameters:
    Lam = lambda * Lref^2 / chi  (= i*Omega, Omega = omega Lref^2/chi)
    L   = l / Lref
    Tt  = tau * lambda           (= i*omega*tau)
    S   = ls / Lref
Temperature is scaled by Lref (source z -> x), heat flux by k.
"""
import numpy as np
from scipy.special import ive
from numpy.polynomial import chebyshev as Ch

# ----------------------------------------------------------------------------
# material data: silicon, 300 K (Peng et al. IJHMT 2025/2026, Tables 1 and 6)
# ----------------------------------------------------------------------------
SI = dict(E=160e9, rho=2330.0, nu=0.22, k=150.0, cE=695.0, alpha=2.6e-6, T0=300.0,
          tau=4.6e-12)
SI['C'] = SI['rho'] * SI['cE']
SI['chi'] = SI['k'] / SI['C']
SI['DeltaE'] = SI['E'] * SI['alpha'] ** 2 * SI['T0'] / SI['C']


def omega_cantilever_circ(a, mode=1, L_over_a=10.0, mat=SI):
    """Euler-Bernoulli cantilever, circular section radius a, length L = L_over_a * a."""
    bL = {1: 1.875104069, 2: 4.694091133, 3: 7.854757438}[mode]
    L = L_over_a * a
    return bL ** 2 / L ** 2 * (a / 2.0) * np.sqrt(mat['E'] / mat['rho'])


def omega_clamped_rect(h, mode=1, L_over_h=10.0, mat=SI):
    """Euler-Bernoulli doubly clamped beam, rectangular section thickness h, L = L_over_h * h."""
    bL = {1: 4.730040745, 2: 7.853204624, 3: 10.99560784}[mode]
    L = L_over_h * h
    return bL ** 2 / L ** 2 * np.sqrt(mat['E'] * h ** 2 / (12.0 * mat['rho']))


def nondim(omega, Lref, l, tau, ls=0.0, mat=SI):
    lam = 1j * omega
    S = np.inf if np.isinf(ls) else ls / Lref
    return dict(Lam=lam * Lref ** 2 / mat['chi'], L=l / Lref, Tt=tau * lam, S=S)


# ----------------------------------------------------------------------------
# Bessel helpers (exponentially scaled -> safe for large complex arguments)
# ----------------------------------------------------------------------------
def rho1(x):
    """I1'(x)/I1(x)"""
    return (ive(0, x) + ive(2, x)) / (2.0 * ive(1, x))


def r21(x):
    """I2(x)/I1(x)"""
    return ive(2, x) / ive(1, x)


def ratio_I1(x_r, x_a):
    """I1(x_r)/I1(x_a) for complex x with Re>0, computed without overflow."""
    return ive(1, x_r) / ive(1, x_a) * np.exp(np.real(x_r) - np.real(x_a))


def ratio_I(nu, x_r, x_a):
    """I_nu(x_r)/I1(x_a)"""
    return ive(nu, x_r) / ive(1, x_a) * np.exp(np.real(x_r) - np.real(x_a))


# ----------------------------------------------------------------------------
# 1D slab (rectangular beam, heat flow across the thickness)
# ----------------------------------------------------------------------------
def g1(x):
    return 12.0 / x ** 2 * (1.0 - 2.0 / x * np.tanh(x / 2.0))


def slab_I(Lam, L, Tt, model, alpha=2.0):
    m = np.sqrt(Lam * (1 + Tt) / (1 + (1 + alpha) * Lam * L ** 2))
    if model == 'S':
        return -1.0 + g1(m)
    if model in ('P', 'consistent'):
        return -1.0 + g1(m) / (1 + (1 + alpha) * Lam * L ** 2)
    raise ValueError(model)


def slab_fd(Lam, L, Tt, N=4000):
    """Independent staggered finite-difference solution of the mixed (q,T) slab problem
    with q(+-1/2) = 0. Returns I."""
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import spsolve
    dz = 1.0 / N
    zn = np.linspace(-0.5, 0.5, N + 1)
    zc = 0.5 * (zn[1:] + zn[:-1])
    nq, nT = N - 1, N
    A = lil_matrix((nq + nT, nq + nT), dtype=complex)
    b = np.zeros(nq + nT, complex)
    for j in range(N):                      # energy in cell j
        r = nq + j
        A[r, nq + j] = Lam
        b[r] = -Lam * zc[j]
        if j + 1 <= N - 1:
            A[r, j] += 1 / dz
        if j >= 1:
            A[r, j - 1] -= 1 / dz
    for i in range(1, N):                   # G-K at interior node i
        r = i - 1
        A[r, r] += (1 + Tt) + 6 * L ** 2 / dz ** 2
        if i + 1 <= N - 1:
            A[r, i] += -3 * L ** 2 / dz ** 2
        if i - 1 >= 1:
            A[r, i - 2] += -3 * L ** 2 / dz ** 2
        A[r, nq + i] += 1 / dz
        A[r, nq + i - 1] -= 1 / dz
    x = spsolve(csr_matrix(A), b)
    T = x[nq:]
    return np.sum(zc * T) * dz * 12.0


def slab_energy_check(Lam, L, Tt, model, n=20001):
    """Return (Qinv_moment/DeltaE, Qinv_dissipation/DeltaE, wall flux q(1/2))."""
    m = np.sqrt(Lam * (1 + Tt) / (1 + 3 * Lam * L ** 2))
    z = np.linspace(-0.5, 0.5, n)
    if model == 'S':
        b = 1.0 / (m * np.cosh(m / 2))
    else:
        b = (1.0 / (1 + 3 * Lam * L ** 2)) / (m * np.cosh(m / 2))
    th = b * np.sinh(m * z) - z
    dth = b * m * np.cosh(m * z) - 1.0
    # heat flux reconstructed from G-K + energy (valid for both models in the bulk)
    q = -((1 + 3 * Lam * L ** 2) * dth + 3 * Lam * L ** 2) / (1 + Tt)
    dq = -Lam * (th + z)
    D = np.trapezoid(np.abs(q) ** 2 + 3 * L ** 2 * np.abs(dq) ** 2, z)
    I = np.trapezoid(z * th, z) * 12.0
    Omega = np.imag(Lam)
    return abs(np.imag(I)), 12.0 * D / Omega, q[-1]


# ----------------------------------------------------------------------------
# 2D disk (circular beam cross-section), azimuthal mode sin(phi)
# ----------------------------------------------------------------------------
def disk_coeffs(Lam, L, Tt, S, model, alpha=2.0):
    """Return dict with m, kappa, U, At (= A I1(m)), Bt (= B I1(kappa)), I."""
    m = np.sqrt(Lam * (1 + Tt) / (1 + (1 + alpha) * Lam * L ** 2))
    U = 1.0 / (1 + Tt)
    out = dict(m=m, U=U, model=model)
    if model == 'S':
        At = -Lam / (m ** 3 * rho1(m))
        Bt, kap = 0.0, None
    elif model == 'P':
        At = -U / (m * rho1(m))
        Bt, kap = 0.0, None
    elif model == 'slip':
        kap = np.sqrt(1 + Tt) / L
        rm, rk = rho1(m), rho1(kap)
        M = np.zeros((2, 2), complex)
        rhs = np.zeros(2, complex)
        M[0] = [m * rm, -1.0]
        rhs[0] = -U
        if np.isinf(S):
            M[1] = [m * rm - 2.0, 2 * kap * rk - kap ** 2 - 1.0]
            rhs[1] = U
        else:
            M[1] = [1.0 + S * (m * rm - 2.0), -kap * rk + S * (2 * kap * rk - kap ** 2 - 1.0)]
            rhs[1] = -U * (1.0 - S)
        At, Bt = np.linalg.solve(M, rhs)
    else:
        raise ValueError(model)
    out.update(kappa=kap, At=At, Bt=Bt)
    out['I'] = -1.0 - 4.0 * m * At * r21(m) / Lam
    return out


def disk_I(Lam, L, Tt, S, model, alpha=2.0):
    return disk_coeffs(Lam, L, Tt, S, model, alpha=alpha)['I']


def disk_fields(c, x):
    """Radial profiles Q_r, Q_phi, their derivatives and Theta (T = Theta sin phi) on x in (0,1]."""
    m, U, At, Bt, kap = c['m'], c['U'], c['At'], c['Bt'], c['kappa']
    I1m = ratio_I(1, m * x, m)          # I1(m x)/I1(m)
    I0m = ratio_I(0, m * x, m)
    I2m = ratio_I(2, m * x, m)
    dI1m = 0.5 * (I0m + I2m)            # I1'(m x)/I1(m)
    ddI1m = -dI1m / (m * x) + (1 + 1 / (m * x) ** 2) * I1m
    A = At
    Qr = U + A * m * dI1m
    Qp = U + A * I1m / x
    dQr = A * m ** 2 * ddI1m
    dQp = -A * I1m / x ** 2 + A * m * dI1m / x
    if c['model'] == 'slip':
        I1k = ratio_I(1, kap * x, kap)
        I0k = ratio_I(0, kap * x, kap)
        I2k = ratio_I(2, kap * x, kap)
        dI1k = 0.5 * (I0k + I2k)
        ddI1k = -dI1k / (kap * x) + (1 + 1 / (kap * x) ** 2) * I1k
        B = Bt
        Qr = Qr - B * I1k / x
        Qp = Qp - B * kap * dI1k
        dQr = dQr + B * I1k / x ** 2 - B * kap * dI1k / x
        dQp = dQp - B * kap ** 2 * ddI1k
    Th = -m ** 2 * A * I1m / c['Lam'] - x
    return Qr, Qp, dQr, dQp, Th


def disk_energy_check(Lam, L, Tt, S, model, n=40001):
    """Q^-1/DeltaE from (i) thermal moment and (ii) total dissipation (bulk + wall)."""
    c = disk_coeffs(Lam, L, Tt, S, model)
    c['Lam'] = Lam
    # graded grid to resolve the Knudsen layer near x = 1
    s = np.linspace(0, 1, n)
    x = 1.0 - (1.0 - s) ** 3
    x[0] = 1e-9
    Qr, Qp, dQr, dQp, Th = disk_fields(c, x)
    div = dQr + Qr / x - Qp / x
    e_rr, e_pp, e_rp = dQr, (Qr - Qp) / x, 0.5 * (dQp - Qp / x + Qr / x)
    bulk = np.pi * np.trapezoid((np.abs(Qr) ** 2 + np.abs(Qp) ** 2) * x, x)
    visc = L ** 2 * np.pi * np.trapezoid(
        (2 * (np.abs(e_rr) ** 2 + np.abs(e_pp) ** 2 + 2 * np.abs(e_rp) ** 2) + np.abs(div) ** 2) * x, x)
    wall = 0.0
    if model == 'slip' and not np.isinf(S) and S > 0:
        wall = L ** 2 / S * np.pi * np.abs(Qp[-1]) ** 2
    Omega = np.imag(Lam)
    I_num = 4.0 * np.trapezoid(x ** 2 * Th, x)
    return dict(moment=abs(np.imag(c['I'])), moment_quad=abs(np.imag(I_num)),
                dissipation=4.0 * (bulk + visc + wall) / (np.pi * Omega),
                wall_flux=Qr[-1], wall_tangential=Qp[-1])


# ----------------------------------------------------------------------------
# independent check: Chebyshev collocation of the radial ODE system
# ----------------------------------------------------------------------------
def cheb(N):
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2.0, np.ones(N - 1), 2.0]) * (-1.0) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return D, x


def _fold(Dfull, M, parity):
    left, right = Dfull[:M, :M], Dfull[:M, M:][:, ::-1]
    return left + right if parity == 'even' else left - right


def _integrate_odd(x, vals_half, N):
    """int_0^1 f dx for an odd function sampled on the positive half of the Chebyshev grid."""
    M = (N + 1) // 2
    full = np.empty(N + 1, complex)
    full[:M] = vals_half
    full[M:] = -vals_half[::-1]
    cr = Ch.chebfit(x, full.real, N)
    ci = Ch.chebfit(x, full.imag, N)
    Fr, Fi = Ch.chebint(cr), Ch.chebint(ci)
    return (Ch.chebval(1.0, Fr) - Ch.chebval(0.0, Fr)) + 1j * (Ch.chebval(1.0, Fi) - Ch.chebval(0.0, Fi))


def disk_collocation(Lam, L, Tt, S, model, N=121):
    """Spectral collocation of the full (Q_r, Q_phi) system (model 'slip') or of the
    scalar temperature equation (model 'S'). Returns I."""
    assert N % 2 == 1
    Dfull, xf = cheb(N)
    D2full = Dfull @ Dfull
    M = (N + 1) // 2
    x = xf[:M]
    if model == 'S':
        m2 = Lam * (1 + Tt) / (1 + 3 * Lam * L ** 2)
        Do, D2o = _fold(Dfull, M, 'odd'), _fold(D2full, M, 'odd')
        A = D2o + np.diag(1 / x) @ Do - np.diag(1 / x ** 2) - m2 * np.eye(M)
        rhs = m2 * x.astype(complex)
        A = A.astype(complex)
        A[0] = Do[0]
        rhs[0] = 0.0
        Th = np.linalg.solve(A, rhs)
        return 4.0 * _integrate_odd(xf, x ** 2 * Th, N)
    if model != 'slip':
        raise ValueError(model)
    De, D2e, Do = _fold(Dfull, M, 'even'), _fold(D2full, M, 'even'), _fold(Dfull, M, 'odd')
    Ri, Ri2, Id = np.diag(1 / x), np.diag(1 / x ** 2), np.eye(M)
    Z = np.zeros((M, M))
    Dmat = np.hstack([De + Ri, -Ri])                     # div q = D(r) sin(phi)  (odd)
    Dp = Do @ Dmat                                       # D'(r)                 (even)
    coef = 2 * L ** 2 + 1.0 / Lam                        # 2 l^2 + chi/lambda
    lapr = np.hstack([D2e + Ri @ De - 2 * Ri2, 2 * Ri2])
    lapp = np.hstack([2 * Ri2, D2e + Ri @ De - 2 * Ri2])
    Er = (1 + Tt) * np.hstack([Id, Z]) - L ** 2 * lapr - coef * Dp
    Ep = (1 + Tt) * np.hstack([Z, Id]) - L ** 2 * lapp - coef * (Ri @ Dmat)
    A = np.vstack([Er, Ep]).astype(complex)
    rhs = np.ones(2 * M, complex)                        # = k (non-dimensional)
    # boundary rows at x = 1 (index 0 of each block)
    A[0] = 0.0
    A[0, 0] = 1.0
    rhs[0] = 0.0                                          # Q_r(1) = 0
    A[M] = 0.0
    if np.isinf(S):
        A[M, M:] = De[0]
        A[M, M] -= 1.0                                    # Q_phi' - Q_phi = 0
    else:
        A[M, M:] = S * De[0]
        A[M, M] += 1.0 - S                                # Q_phi + S (Q_phi' - Q_phi) = 0
    rhs[M] = 0.0
    sol = np.linalg.solve(A, rhs)
    Dv = Dmat @ sol
    Th = -Dv / Lam - x
    return 4.0 * _integrate_odd(xf, x ** 2 * Th, N)


# ----------------------------------------------------------------------------
# Knudsen-layer wall model (Wentzell-type boundary condition), disk closed form
#   (chi + 3 lam l^2) (dT/dn + delta d2T/ds2) = -3 lam l^2 (dtheta/dn + delta d2theta/ds2)
#   delta = l / (beta (1 + Cs beta)),  beta = sqrt(1 + tau lam),  ls = Cs l
# On the unit disk with theta = z = r sin(phi):  d/dn z = sin, d2/ds2 z = -sin, d2/ds2 T = -T.
# ----------------------------------------------------------------------------
def wall_delta(L, Tt, Cs):
    beta = np.sqrt(1 + Tt)
    if np.isinf(Cs):
        return 0.0
    return L / (beta * (1 + Cs * beta))


def disk_I_wall(Lam, L, Tt, Cs, alpha=2.0):
    m = np.sqrt(Lam * (1 + Tt) / (1 + (1 + alpha) * Lam * L ** 2))
    d = wall_delta(L, Tt, Cs)
    At = -Lam * (1 - d) / ((1 + (1 + alpha) * Lam * L ** 2) * m ** 2 * (m * rho1(m) - d))
    return -1.0 - 4.0 * m * At * r21(m) / Lam


def rect_sidewall_F(m):
    """F(m) = -1/m^2 + 3 tanh(m/2)/m^3 - 1/(2 m^2 cosh^2(m/2))  (side-wall correction kernel)."""
    m = np.asarray(m, complex)
    sech2 = 1.0 / np.cosh(m / 2) ** 2
    return -1.0 / m ** 2 + 3.0 * np.tanh(m / 2) / m ** 3 - 0.5 * sech2 / m ** 2


def rect_I_wall(Lam, L, Tt, B, Cs, alpha=2.0):
    """Rectangle of width B (thickness 1): consistent slab + first-order side-wall Knudsen correction
    I = I_slab + 24 delta F(m) / ((1 + 3 Lam L^2) B)."""
    a = 1 + (1 + alpha) * Lam * L ** 2
    m = np.sqrt(Lam * (1 + Tt) / a)
    return slab_I(Lam, L, Tt, 'consistent', alpha=alpha) + 24.0 * wall_delta(L, Tt, Cs) * rect_sidewall_F(m) / (a * B)


def rect_I_eff(Lam, L, Tt, B, Cs, alpha=2.0):
    """Effective-diffusivity slab for a B x 1 rectangle: side-wall Knudsen layers remove a fraction
    eps = 2 delta / B of the z-flux, so  m_e^2 = m^2 / (1 - eps),  I = -1 + g1(m_e)/(1 + 3 Lam L^2).
    Its first-order expansion in eps is rect_I_wall (exact first-order perturbation of the W model)."""
    a = 1 + (1 + alpha) * Lam * L ** 2
    m = np.sqrt(Lam * (1 + Tt) / a)
    eps = 2.0 * wall_delta(L, Tt, Cs) / B
    me = m / np.sqrt(1 - eps)
    g1 = 12.0 / me ** 2 * (1 - 2.0 / me * np.tanh(me / 2))
    return -1.0 + g1 / a
