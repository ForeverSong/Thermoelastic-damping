"""Table: wall model W vs exact consistent solution on the disk.
Error measure |I_model - I| / |1 + I| (deviation of the thermal response from the adiabatic value)."""
import numpy as np, os, json, gkted as g
Tt = 1e-3j; rows = []
sci = lambda x: f"{x:.1e}".replace('e-0', 'e-').replace('e+0', 'e')
with open(os.path.join('..', 'tables', 'tab_wall.tex'), 'w') as f:
    for L in (0.3, 0.1, 0.03, 0.01, 0.003, 0.001):
        cells = [f"{L:g}"]
        for Om, Cs in ((3.39, 0.0), (30.0, 0.0), (3.39, 1.0)):
            Lam = 1j * Om
            ex = g.disk_I(Lam, L, Tt, Cs * L, 'slip')
            e = lambda I: abs(I - ex) / abs(1 + ex)
            r = dict(L=L, Om=Om, Cs=Cs, S=e(g.disk_I(Lam, L, Tt, 0, 'S')), P=e(g.disk_I(Lam, L, Tt, 0, 'P')),
                     W=e(g.disk_I_wall(Lam, L, Tt, Cs)))
            rows.append(r)
            cells += [sci(r['S']), sci(r['P']), sci(r['W'])] if Cs == 0 else [sci(r['P']), sci(r['W'])]
        f.write(' & '.join(cells) + ' \\\\\n')
worst = max(np.imag(g.disk_I_wall(1j * Om, L, 1j * wt, Cs)) for Cs in (0.0, 0.5, 1.0, 3.0)
            for Om in np.logspace(-2, 4, 121) for L in np.logspace(-4, 0, 81) for wt in (0.0, 1e-3, 1e-2, 0.1))
json.dump(dict(rows=rows, max_ImI_W=float(worst)), open('../wall_table.json', 'w'), indent=1)
print(open('../tables/tab_wall.tex').read(), 'max Im I_W =', worst)
