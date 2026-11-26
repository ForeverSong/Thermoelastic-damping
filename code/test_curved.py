import numpy as np, time
import fem_mesh as fm, fem_mixed_curved as mc, gkted as g
Tt = 1e-3j
# sanity 1: rectangle (straight) must reproduce the slab as before
Om, L = 10.0, 0.1; Lam = 1j * Om; ex = g.slab_I(Lam, L, Tt, 'consistent')
for n in (8, 16, 32):
    S = mc.build(fm.rect_mesh(1.0, 1.0, n, n)); A = mc.assemble(S, L, marker=lambda x: 'free' if abs(abs(x[0]) - 0.5) < 1e-9 else 'wall')
    r0 = mc.solve(A, Lam, L, Tt, 0.0)
    print(f"rect n={n}: err no-slip walls = {abs(r0['I']-ex)/abs(ex):.2e}  identity={abs(abs(np.imag(r0['I'])) - r0['D']/(Om*A['zz']))/abs(np.imag(r0['I'])):.1e}")
# disk with curved elements
Om, L = 3.39, 0.1; Lam = 1j * Om
ex = {S_: g.disk_I(Lam, L, Tt, S_, 'slip') for S_ in (0.0, L, np.inf)}
for h in (0.2, 0.1, 0.05, 0.025):
    t0 = time.time()
    S = mc.build(fm.disk_mesh(h, grade=(0.4, (L / 4) * (h / 0.2))))
    A = mc.assemble(S, L)
    errs, ids = [], []
    for S_ in (0.0, L, np.inf):
        r = mc.solve(A, Lam, L, Tt, S_)
        errs.append(abs(r['I'] - ex[S_]) / abs(ex[S_]))
        ids.append(abs(abs(np.imag(r['I'])) - r['D'] / (Om * A['zz'])) / abs(np.imag(r['I'])))
    print(f"disk h={h:5.3f} nt={S['nt']:6d}: no-slip {errs[0]:.2e}  slip {errs[1]:.2e}  perfect {errs[2]:.2e}   identity max {max(ids):.1e}  ({time.time()-t0:.0f}s)")
