import numpy as np, json, time
import fem_mesh as fm, fem_mixed as mx, fem_scalar as fs, gkted as g
Tt0 = 1e-3   # T-hat = i * Om * Tt0 * ... keep tau-effect small and proportional to Om
q = lambda I: abs(np.imag(I))
out = []
for B in (1, 2, 4):
    As = fs.assemble(fm.rect_mesh(B, 1.0, 24 * B, 24))
    for L in (0.05, 0.1):
        mesh = fm.rect_mesh(B, 1.0, 12 * B, 12, wall_h=L / 6)
        S = mx.build(mesh)
        side_free = lambda x: 'free' if abs(abs(x[0]) - B / 2) < 1e-9 else 'wall'
        Aw = mx.assemble(S, L); Af = mx.assemble(S, L, marker=side_free)
        for Om in np.logspace(-0.5, 2.5, 13):
            Lam = 1j * Om; Tt = 1j * Om * Tt0 * L ** 2
            I1 = g.slab_I(Lam, L, Tt, 'consistent')
            dT = mx.solve(Aw, Lam, L, Tt, 0.0)['I'] - mx.solve(Af, Lam, L, Tt, 0.0)['I']
            IW = fs.solve(As, Lam, L, Tt, 'W', 0.0)[0]
            Ia = g.rect_I_wall(Lam, L, Tt, B, 0.0)
            IS = g.slab_I(Lam, L, Tt, 'S')
            out.append(dict(B=B, L=L, Om=Om, Q1=q(I1), true=float(np.imag(dT) / np.imag(I1)),
                            W=float((np.imag(IW) - np.imag(I1)) / np.imag(I1)),
                            asym=float((np.imag(Ia) - np.imag(I1)) / np.imag(I1)),
                            S=float((np.imag(IS) - np.imag(I1)) / np.imag(I1)),
                            reTrue=float(np.real(dT)), reW=float(np.real(IW - I1))))
            r = out[-1]
            json.dump(out, open("../rect_omega.json", "w"), indent=1)
            print(f"B={B} L={L} Om={Om:8.2f}: dQ/Q true={r['true']:+.2e} W={r['W']:+.2e} asym={r['asym']:+.2e}  | S-leak={r['S']:+.2e}", flush=True)

