"""Wall model on rectangles: low-frequency side-wall correction vs delta (manuscript Table: near-resonances),
and scan of the effective slip length implied by the scalar closure on the disk (Remark on the active wall)."""
import numpy as np, os, json
import fem_mesh as fm, fem_scalar as fs, fem_mixed as mx, gkted as g
from gkted import rho1
out = dict(rows=[])
Lam, T0 = 0.316j, 0.0
with open(os.path.join('..', 'tables', 'tab_reson.tex'), 'w') as f:
    As = {B: (fs.assemble(fm.rect_mesh(B, 1.0, 24 * B, 24)), fs.assemble(fm.rect_mesh(B, 1.0, 36 * B, 36))) for B in (1, 2)}
    for L in (0.03, 0.05, 0.06, 0.07, 0.08, 0.085, 0.09, 0.095, 0.1, 0.13):
        I1 = np.imag(g.slab_I(Lam, L, T0, 'consistent')); cells = [f"{L:g}"]
        for B in (1, 2):
            c = lambda I: B * (np.imag(I) - I1) / I1 / (2 * L)
            free = lambda x, B=B: 'free' if abs(abs(x[0]) - B / 2) < 1e-9 else 'wall'
            S = mx.build(fm.rect_mesh(B, 1.0, 16 * B, 16, wall_h=L / 8)); Aw = mx.assemble(S, L); Af = mx.assemble(S, L, marker=free)
            ref = B * (np.imag(mx.solve(Aw, Lam, L, T0, 0.0)['I']) / np.imag(mx.solve(Af, Lam, L, T0, 0.0)['I']) - 1) / (2 * L)
            w1, w2 = c(fs.solve(As[B][0], Lam, L, T0, 'W', 0.0)[0]), c(fs.solve(As[B][1], Lam, L, T0, 'W', 0.0)[0])
            ef = c(g.rect_I_eff(Lam, L, T0, B, 0.0))
            out['rows'].append(dict(B=B, delta=L, mixed=float(ref), W24=float(w1), W36=float(w2), eff=float(ef)))
            cells += [f"{ref:.3f}", f"{w1:.3f}", f"{w2:.3f}", f"{ef:.3f}"]
        f.write(' & '.join(cells) + ' \\\\\n'); print(cells, flush=True)
# scalar closure on the disk: effective slip length of the vortical completion
worstS, worstW, n = -np.inf, -np.inf, 0
for Om in np.logspace(-1, 2, 31):
    for L in np.logspace(-2, np.log10(0.5), 25):
        for wt in (0.0, 1e-2, 0.1):
            Lm, Tt = 1j * Om, 1j * wt
            m = np.sqrt(Lm * (1 + Tt) / (1 + 3 * Lm * L**2)); U = 1 / (1 + Tt); kap = np.sqrt(1 + Tt) / L
            At = -Lm / (m**3 * rho1(m)); Bt = U + At * m * rho1(m)
            qphi = U + At - Bt * kap * rho1(kap); trac = -U + At * (m * rho1(m) - 2) + Bt * (2 * kap * rho1(kap) - (kap**2 + 1))
            worstS = max(worstS, np.real(-qphi / trac) / L); worstW = max(worstW, -np.real(np.conj(qphi) * trac)); n += 1
out['scalar_completion'] = dict(cases=n, max_Re_Seff_over_L=float(worstS), max_wall_power=float(worstW))
print(out['scalar_completion'])
json.dump(out, open('../wall_resonance.json', 'w'), indent=1)
