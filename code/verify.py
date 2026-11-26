"""Verification of the closed-form solutions in gkted.py.
1. slab: closed form vs staggered finite differences of the mixed (q,T) system
2. disk: closed form vs Chebyshev collocation (scalar model and full slip family)
3. energy identity: Q^-1 from the thermal moment == Q^-1 from total dissipation
   (holds for consistent models, fails for the scalar model -> boundary leak)
"""
import numpy as np
import gkted as g

np.set_printoptions(precision=6)
print("Delta_E (Si, 300 K) =", g.SI['DeltaE'])

print("\n=== 1. slab: closed form vs finite differences (N = 4000) ===")
print(f"{'Omega':>8} {'L':>6} {'closed I':>26} {'FD I':>26} {'rel.err':>9}")
for Om, L in [(5.0, 0.08), (30.0, 0.08), (30.0, 0.3), (200.0, 0.05), (2.0, 0.5)]:
    Lam, Tt = 1j * Om, 1j * 1e-3
    Ic = g.slab_I(Lam, L, Tt, 'consistent')
    If = g.slab_fd(Lam, L, Tt)
    print(f"{Om:8.1f} {L:6.2f} {Ic:26.8f} {If:26.8f} {abs(Ic - If) / abs(Ic):9.1e}")

print("\n=== 2. disk: closed form vs Chebyshev collocation (N = 121) ===")
cases = [(3.4, 0.1, 0.0), (3.4, 0.1, 0.1), (3.4, 0.1, np.inf), (20.0, 0.3, 0.3),
         (0.5, 0.05, 0.05), (100.0, 0.02, 0.0), (3.4, 0.6, 0.6)]
print(f"{'Omega':>7} {'L':>5} {'S':>5}  model  {'closed I':>26} {'colloc I':>26} {'rel.err':>9}")
for Om, L, S in cases:
    Lam, Tt = 1j * Om, 1j * 1e-3
    for model in ('S', 'slip'):
        Ic = g.disk_I(Lam, L, Tt, S, model)
        Ik = g.disk_collocation(Lam, L, Tt, S, model, N=121)
        print(f"{Om:7.1f} {L:5.2f} {S:5.2f}  {model:5s}  {Ic:26.8f} {Ik:26.8f} {abs(Ic - Ik) / abs(Ic):9.1e}")

print("\n=== 3a. energy identity, slab: |Im I| vs 12*D/Omega ===")
for model in ('consistent', 'S'):
    for Om, L in [(10.0, 0.1), (30.0, 0.3)]:
        mo, di, qw = g.slab_energy_check(1j * Om, L, 1j * 1e-3, model)
        print(f"{model:10s} Omega={Om:5.1f} L={L:4.2f}: moment={mo:.6f} dissipation={di:.6f} "
              f"mismatch={abs(mo - di) / mo:8.1e}  wall flux q(1/2)={qw:.4f}")

print("\n=== 3b. energy identity, disk: |Im I| vs 4*D/(pi*Omega) ===")
for model, S in [('slip', 0.0), ('slip', 0.1), ('slip', np.inf), ('S', 0.0), ('P', 0.0)]:
    for Om, L in [(3.4, 0.1), (20.0, 0.3)]:
        r = g.disk_energy_check(1j * Om, L, 1j * 1e-3, S, model)
        print(f"{model:4s} S={S:4.2f} Omega={Om:5.1f} L={L:4.2f}: moment={r['moment']:.6f} "
              f"(quad {r['moment_quad']:.6f}) dissipation={r['dissipation']:.6f} "
              f"mismatch={abs(r['moment'] - r['dissipation']) / r['moment']:8.1e}  "
              f"q_n(wall)={abs(r['wall_flux']):.2e}")
