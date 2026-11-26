import numpy as np
import fem_mesh as fm, fem_mixed as mx, fem_scalar as fs, gkted as g
Tt = 1e-3j; Om = np.pi ** 2; Lam = 1j * Om; B = 2; Cs = 0.0
A = fs.assemble(fm.rect_mesh(B, 1.0, 32 * B, 32))
side_free = lambda x: 'free' if abs(abs(x[0]) - B / 2) < 1e-9 else 'wall'
for L in (0.0025, 0.005, 0.01, 0.02, 0.05):
    I1 = g.slab_I(Lam, L, Tt, 'consistent')
    dW = fs.solve(A, Lam, L, Tt, 'W', Cs)[0] - I1
    dA = g.rect_I_wall(Lam, L, Tt, B, Cs) - I1
    mesh = fm.rect_mesh(B, 1.0, 32 * B, 32, wall_h=L / 8)
    S = mx.build(mesh)
    Aw = mx.assemble(S, L); Af = mx.assemble(S, L, marker=side_free)
    dT = mx.solve(Aw, Lam, L, Tt, Cs * L)['I'] - mx.solve(Af, Lam, L, Tt, Cs * L)['I']
    print(f"L={L:6.4f} dI_true={dT:.3e}  dI_W={dW:.3e}  dI_asym={dA:.3e}   ratio true/asym={dT/dA:.3f}  W/asym={dW/dA:.3f}", flush=True)
