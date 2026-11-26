import numpy as np, time
import fem_mesh as fm, fem_scalar as fs, gkted as g
Lam, L, Tt = 3.39j, 0.1, 1e-3j
ex = {'S': g.disk_I(Lam, L, Tt, 0, 'S'), 'P': g.disk_I(Lam, L, Tt, 0, 'P'), 'W': g.disk_I_wall(Lam, L, Tt, 0.0)}
print('exact', ex)
prev = None
for h in (0.4, 0.2, 0.1, 0.05, 0.025):
    t0 = time.time()
    A = fs.assemble(fm.disk_mesh(h))
    out = []
    for mdl in ('S', 'P', 'W'):
        I, _ = fs.solve(A, Lam, L, Tt, mdl, Cs=0.0)
        out.append(abs(I - ex[mdl]) / abs(ex[mdl]))
    print(f"h={h:6.3f} ndof={A['K'].shape[0]:6d} area err={abs(A['area']-np.pi):.1e}  errS={out[0]:.2e} errP={out[1]:.2e} errW={out[2]:.2e}  ({time.time()-t0:.1f}s)")
