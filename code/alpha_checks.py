"""General coefficient alpha of grad(div q):  tau q_t + q = -k grad T + l^2 (lap q + alpha grad div q).
alpha = 2: Guyer-Krumhansl;  alpha = 1/3: kinetic form at ordinary temperatures (Guo & Wang 2018).
Checks of the alpha-general closed forms and solvers."""
import numpy as np, json
import gkted as g, fem_mesh as fm, fem_mixed as mx, fem_mixed_curved as mc, fem_scalar as fs
al = 1.0 / 3.0; Tt = 1e-3j; out = {}
print("== 1. slab: alpha enters only through (1+alpha) L^2")
d = max(abs(g.slab_I(1j * Om, Kn / np.sqrt(5), 1j * Om * Kn**2 / 3, mdl, alpha=al) - g.slab_I(1j * Om, np.sqrt(4 / 45) * Kn, 1j * Om * Kn**2 / 3, mdl))
        for Om in (0.3, 3.0, 30.0) for Kn in (0.05, 0.3) for mdl in ('S', 'consistent'))
print("   max |I(alpha=1/3, L=Kn/sqrt5) - I(alpha=2, L=sqrt(4/45)Kn)| =", d); out['slab_equiv'] = float(d)
print("== 2. disk, alpha=1/3: curved mixed FE (weak form) vs closed form (strong form), Omega=3.39, L=0.1")
Lam, L = 3.39j, 0.1; ex = {s: g.disk_I(Lam, L, Tt, s, 'slip', alpha=al) for s in (0.0, L, np.inf)}; rows = []
for h in (0.2, 0.1, 0.05):
    S = mc.build(fm.disk_mesh(h, grade=(0.4, (L / 4) * (h / 0.2)))); A = mc.assemble(S, L); row = dict(h=h)
    for key, s in (('noslip', 0.0), ('slip', L), ('perfect', np.inf)):
        r = mc.solve(A, Lam, L, Tt, s, alpha=al); row[key] = float(abs(r['I'] - ex[s]) / abs(ex[s]))
        row[key + '_id'] = float(abs(abs(np.imag(r['I'])) - r['D'] / (r['Omega'] * A['zz'])) / abs(np.imag(r['I'])))
        nT = np.sqrt(np.sum(np.abs(r['T'])**2 * A['area'])); row[key + '_stab'] = float(nT / np.sqrt(A['zz']))
    rows.append(row); print("  ", {k: (f"{v:.2e}" if k != 'h' else v) for k, v in row.items()}, flush=True)
out['disk_conv'] = rows
print("   exact values: ", {str(k): complex(np.round(v, 6)) for k, v in ex.items()})
print("== 3. positivity of the viscous form 2(eps,eps) + (alpha-1)(div,div) + j_h for alpha=1/3 (eta=10)")
for name, S_, mod in (("curved disk h=0.2", mc.build(fm.disk_mesh(0.2, grade=(0.4, 0.025))), mc), ("rectangle 2x1 graded", mx.build(fm.rect_mesh(2, 1.0, 16, 8, wall_h=0.005)), mx)):
    A = mod.assemble(S_, 0.1); f = A['free']
    for ns in (False, True):
        M = (A['Vq'] + (al - 2) * A['Vd'] + A['Jq'] + (A['Bn'] if ns else 0 * A['Bn']))[f][:, f].toarray(); w = np.linalg.eigvalsh(0.5 * (M + M.T))
        print(f"   {name:22s} {'no-slip' if ns else 'perfect slip'}: lambda_min = {w[0]:+.3e}"); out[f'eig_{name}_{ns}'] = float(w[0])
print("== 4. P2 scalar solver with alpha=1/3 on the disk: scalar closure and wall law vs closed forms")
As = fs.assemble(fm.disk_mesh(0.05))
for mdl, exv in (('S', g.disk_I(Lam, L, Tt, 0, 'S', alpha=al)), ('W', g.disk_I_wall(Lam, L, Tt, 0.0, alpha=al))):
    I = fs.solve(As, Lam, L, Tt, mdl, 0.0, alpha=al)[0]; print(f"   {mdl}: rel. err {abs(I - exv) / abs(exv):.1e}")
print("   wall law vs exact no-slip (alpha=1/3):", abs(g.disk_I_wall(Lam, L, Tt, 0.0, alpha=al) - ex[0.0]) / abs(1 + ex[0.0]))
print("== 5. rectangle b=2h, alpha=1/3, no-slip: side-wall correction (b/h) dQ/Q, mixed FE vs effective-diffusivity formula")
B, Lr = 2, 0.05; free = lambda x: 'free' if abs(abs(x[0]) - B / 2) < 1e-9 else 'wall'
S = mx.build(fm.rect_mesh(B, 1.0, 16 * B, 16, wall_h=Lr / 8)); Aw = mx.assemble(S, Lr); Af = mx.assemble(S, Lr, marker=free)
for Om in (0.316, 316.0):
    Lm = 1j * Om; I1 = np.imag(g.slab_I(Lm, Lr, 0.0, 'consistent', alpha=al))
    fe = B * (np.imag(mx.solve(Aw, Lm, Lr, 0.0, 0.0, alpha=al)['I']) / np.imag(mx.solve(Af, Lm, Lr, 0.0, 0.0, alpha=al)['I']) - 1)
    slab_err = abs(mx.solve(Af, Lm, Lr, 0.0, 0.0, alpha=al)['I'] - g.slab_I(Lm, Lr, 0.0, 'consistent', alpha=al))
    ef = B * (np.imag(g.rect_I_eff(Lm, Lr, 0.0, B, 0.0, alpha=al)) - I1) / I1
    print(f"   Om={Om:7.2f}: mixed FE {fe:+.4f}   formula {ef:+.4f}   (FE with free sides vs slab closed form: {slab_err:.1e})")
json.dump(out, open('../alpha_checks.json', 'w'), indent=1)
