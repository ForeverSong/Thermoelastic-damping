"""Peak inverse quality factor of the 2nd cantilever mode (circular Si beam, L = 10 a)
versus the nonlocal length l, for each boundary model. Writes tables/tab_peaks.tex."""
import os, json
import numpy as np
from scipy.optimize import minimize_scalar
import gkted as g

DE, tau = g.SI['DeltaE'], g.SI['tau']
HERE = os.path.dirname(os.path.abspath(__file__))


def Q(a, l, mode, model, ls=0.0):
    om = g.omega_cantilever_circ(a, mode)
    p = g.nondim(om, a, l, tau, ls)
    return DE * abs(np.imag(g.disk_I(p['Lam'], p['L'], p['Tt'], p['S'], model)))


def peak(l, model, ls):
    r = minimize_scalar(lambda la: -Q(10 ** la, l, 2, model, ls),
                        bounds=(np.log10(0.08e-6), np.log10(3e-6)), method='bounded',
                        options={'xatol': 1e-6})
    return -r.fun, 10 ** r.x * 1e6


out = {}
lines = []
for l_nm in (40, 80, 120):
    l = l_nm * 1e-9
    row = [f'{l_nm}']
    for key, model, ls in (('S', 'S', 0.0), ('noslip', 'slip', 0.0), ('slip', 'slip', l),
                           ('perfect', 'slip', np.inf)):
        q, a = peak(l, model, ls)
        out[f'{key}_{l_nm}'] = (q, a)
        row.append(f'{q*1e5:.2f} ({a:.2f})')
    lines.append(' & '.join(row) + ' \\\\')
qf, af = peak(1e-15, 'S', 0.0)
out['fourier'] = (qf, af)
with open(os.path.join(HERE, '..', 'tables', 'tab_peaks.tex'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
with open(os.path.join(HERE, '..', 'peaks.json'), 'w') as f:
    json.dump(out, f, indent=1)
for k, v in out.items():
    print(k, f'{v[0]:.4e}', f'{v[1]:.3f} um')
for key in ('S', 'noslip', 'slip', 'perfect'):
    print(key, 'change 40->120 nm:', f"{100*(out[key+'_120'][0]/out[key+'_40'][0]-1):+.1f}%")
