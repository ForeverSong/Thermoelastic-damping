"""Independent re-checks of the theory (audit, 2026-11-25)."""
import numpy as np, gkted as g
np.set_printoptions(precision=6, suppress=False)
print("== 1. Fourier slab vs Lifshitz-Roukes closed form (independent formula)")
def LR(Om):
    xi = np.sqrt(Om / 2)
    return 6 / xi**2 - 6 / xi**3 * (np.sinh(xi) + np.sin(xi)) / (np.cosh(xi) + np.cos(xi))
for Om in (0.5, 3.0, 10.0, 50.0, 300.0):
    print(f"  Om={Om:6.1f}  -Im I={-np.imag(g.slab_I(1j*Om, 0.0, 0.0, 'S')):.12f}  LR={LR(Om):.12f}")

print("== 2. Consistent slab = G-K solved in the evenly extended (specular) domain: mode sum vs closed form")
def modesum(Lam, L, Tt, N=400000):
    k = (2 * np.arange(N) + 1) * np.pi
    r = -Lam * (1 + Tt + 3 * L**2 * k**2) / (Lam * (1 + Tt + 3 * L**2 * k**2) + k**2)
    rinf = -3 * Lam * L**2 / (1 + 3 * Lam * L**2)          # limit of r_n; sum 96/k^4 = 1
    return rinf + 96 * np.sum((r - rinf) / k**4)
for Om, L in ((1.0, 0.1), (10.0, 0.1), (30.0, 0.3), (200.0, 0.05)):
    Lam, Tt = 1j * Om, 1e-3j
    a, b, c = modesum(Lam, L, Tt), g.slab_I(Lam, L, Tt, 'consistent'), g.slab_I(Lam, L, Tt, 'S')
    print(f"  Om={Om:5.1f} L={L}: modesum={a:.10f}  consistent={b:.10f}  |diff|={abs(a-b):.1e}   (scalar closure differs by {abs(a-c):.2e})")

print("== 3. Scalar closure on the disk: can a vortical flux cancel the leak? implied slip length S_eff")
from gkted import rho1
def scalar_completion(Lam, L, Tt):
    m = np.sqrt(Lam * (1 + Tt) / (1 + 3 * Lam * L**2)); U = 1 / (1 + Tt); kap = np.sqrt(1 + Tt) / L
    At = -Lam / (m**3 * rho1(m))                 # scalar closure, dT/dr(1) = 0
    Bt = U + At * m * rho1(m)                    # vortical part that makes q_r(1) = 0
    qphi = U + At - Bt * kap * rho1(kap)
    trac = -U + At * (m * rho1(m) - 2) + Bt * (2 * kap * rho1(kap) - (kap**2 + 1))
    return Bt, qphi, trac, -qphi / trac
for Om, L in ((0.5, 0.1), (3.39, 0.1), (3.39, 0.3), (20.0, 0.3), (50.0, 0.05)):
    Bt, qphi, trac, S = scalar_completion(1j * Om, L, 1e-3j)
    wall = -L**2 * np.real(np.conj(qphi) * trac) * np.pi          # wall term of the energy identity (per unit amplitude^2, int cos^2 = pi)
    print(f"  Om={Om:5.2f} L={L}: leak B~={Bt:.3e}  S_eff={S:.4f}  wall power term={wall:+.3e} ({'dissipative' if wall>=0 else 'INJECTS power'})")

print("== 4. Wall model positivity: where does Im I_W reach zero / change sign?")
worst, arg = -np.inf, None
for Cs in (0.0, 0.5, 1.0, 3.0):
    for Om in np.logspace(-2, 4, 121):
        for L in np.logspace(-4, 0, 81):
            for wt in (0.0, 1e-3, 1e-2, 0.1):
                d = g.wall_delta(L, 1j * wt, Cs)
                if abs(d) < 0.5:
                    v = np.imag(g.disk_I_wall(1j * Om, L, 1j * wt, Cs)) / Om
                    if v > worst: worst, arg = v, (Cs, Om, L, wt)
print("  max of Im(I_W)/Omega over the grid restricted to |delta|<0.5:", worst, arg)
for L in (0.9, 1.0, 1.1, 1.5):
    print(f"  no-slip, Om=1, L={L}: delta={g.wall_delta(L,0,0):.2f}  Im I_W={np.imag(g.disk_I_wall(1j, L, 0.0, 0.0)):+.4e}   exact Im I={np.imag(g.disk_I(1j, L, 0.0, 0.0, 'slip')):+.4e}")
