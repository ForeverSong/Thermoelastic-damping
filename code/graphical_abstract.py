"""Graphical abstract (13 cm x 5.2 cm, aspect 2.5:1).

Every panel is computed from the models and data of the paper; nothing is drawn by hand.
  left   : heat-flux lines on the circular section (exact solutions, Lambda = 3.39i, l/a = 0.1),
           temperature-only closure versus flux boundary conditions (no-slip)
  centre : relative error of Q^-1 against the gray phonon Boltzmann solution, slab, Omega = 3
           (read from ../bte_results.json, written by bte_results.py)
  right  : Q^-1 of a silicon cantilever of circular section, first mode, l = 120 nm
Writes ../figures/graphical_abstract.pdf, .png and .tiff (600 dpi, 3070 x 1228 pixels).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

import gkted as g

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, '..', 'figures')
BLUE, ORANGE = '#257de0', '#f47230'          # existing temperature-only closure / flux boundary conditions
GREY, INK, INK2, GRID = '#8c8c8c', '#191919', '#666666', '#e6e6e6'
DASH = chr(0x2013)

sans = 'Liberation Sans' if any(f.name == 'Liberation Sans' for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
plt.rcParams.update({
    'font.family': sans, 'mathtext.fontset': 'custom', 'mathtext.rm': sans, 'mathtext.it': sans + ':italic',
    'mathtext.bf': sans + ':bold', 'mathtext.cal': sans, 'font.size': 6.6, 'axes.labelsize': 6.6,
    'xtick.labelsize': 6.0, 'ytick.labelsize': 6.0, 'axes.edgecolor': INK2, 'axes.linewidth': 0.5,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
    'pdf.fonttype': 42})

Tt = 1e-3j


def disk_flux(Lam, L, S, model, n=241):
    """Imaginary part of the heat flux of the exact disk solution on a Cartesian grid."""
    c = g.disk_coeffs(Lam, L, Tt, S, model); c['Lam'] = Lam
    y = np.linspace(-1, 1, n); Y, Z = np.meshgrid(y, y)
    R = np.hypot(Y, Z); PH = np.arctan2(Z, Y); out = R > 1.0
    r = np.clip(R, 1e-5, 1.0).ravel()
    Qr, Qp = [np.asarray(a).reshape(R.shape) for a in g.disk_fields(c, r)[:2]]
    qr, qp = Qr * np.sin(PH), Qp * np.cos(PH)
    qy, qz = qr * np.cos(PH) - qp * np.sin(PH), qr * np.sin(PH) + qp * np.cos(PH)
    wall = g.disk_fields(c, np.array([1.0]))[0][0]            # normal-flux amplitude at the wall
    ma = lambda a: np.ma.array(np.imag(a), mask=out)
    return Y, Z, ma(qy), ma(qz), wall


def qinv_disk(a, l, key, mode=1):
    mat = g.SI
    om = g.omega_cantilever_circ(a, mode, 10.0)
    if key == 'fourier':
        p = g.nondim(om, a, 0.0, 0.0)
        return mat['DeltaE'] * abs(np.imag(g.disk_I(p['Lam'], 1e-12, 0.0, 0.0, 'S')))
    ls = {'S': 0.0, 'noslip': 0.0, 'slip': l, 'perfect': np.inf}[key]
    p = g.nondim(om, a, l, mat['tau'], ls)
    return mat['DeltaE'] * abs(np.imag(g.disk_I(p['Lam'], p['L'], p['Tt'], p['S'], 'S' if key == 'S' else 'slip')))


def clean_log_axes(ax):
    ax.set_xscale('log'); ax.set_yscale('log'); ax.minorticks_off()
    ax.grid(True, which='major', color=GRID, lw=0.4); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


W, H = 13.0 / 2.54, 5.2 / 2.54
fig = plt.figure(figsize=(W, H))
fig.text(0.5, 0.945, 'Guyer' + DASH + 'Krumhansl thermoelastic damping: the heat flux needs its own boundary conditions',
         fontsize=7.6, fontweight='bold', va='center', ha='center')
fig.add_artist(plt.Line2D([0.012, 0.988], [0.895, 0.895], color=GRID, lw=0.6))
TITLE_Y = 0.825

# ------------------------------------------------------------------ left: flux lines on the section
th = np.linspace(0, 2 * np.pi, 400)
Lam, L = 3.39j, 0.1
panels = (('S', BLUE, 'Existing closure', r'$\partial T/\partial n=0$', 'heat crosses the wall'),
          ('slip', ORANGE, 'This work', r'$\mathbf{q}\cdot\mathbf{n}=0$ and slip', 'the wall is adiabatic'))
X0, DX, AW = 0.008, 0.2, 0.198
for j, (model, col, head, cond, note) in enumerate(panels):
    ax = fig.add_axes([X0 + DX * j, 0.135, AW, AW * W / H])
    Y, Z, qy, qz, wall = disk_flux(Lam, L, 0.0, model)
    ax.fill(np.cos(th), np.sin(th), color='#f4f6f9', lw=0)
    ax.streamplot(Y, Z, qy, qz, density=0.75, color=INK2, linewidth=0.45, arrowsize=0.45)
    ax.plot(np.cos(th), np.sin(th), color=col, lw=1.8, solid_capstyle='round')
    if model == 'S':
        a = np.linspace(0, 2 * np.pi, 20, endpoint=False) + np.pi / 20
        qn = np.imag(wall) * np.sin(a)
        ax.quiver(np.cos(a), np.sin(a), qn * np.cos(a), qn * np.sin(a), color=col, scale=1.25, width=0.017, zorder=5)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    xc = X0 + DX * j + AW / 2
    fig.text(xc, TITLE_Y, head, ha='center', fontsize=6.8, fontweight='bold')
    fig.text(xc, 0.745, cond, ha='center', fontsize=6.6)
    fig.text(xc, 0.075, note, ha='center', fontsize=6.4, color=INK2)

# ------------------------------------------------------------------ centre: kinetic validation
B = json.load(open(os.path.join(HERE, '..', 'bte_results.json')))
rows = sorted((r for r in B['err'] if abs(r['Om'] - 3.0) < 1e-9), key=lambda r: r['Kn'])
kn = np.array([r['Kn'] for r in rows])
ax = fig.add_axes([0.478, 0.215, 0.19, 0.53])
for key, col, mk in (('scalar', BLUE, 'o'), ('consistent', ORANGE, 's')):
    e = np.array([r[key] for r in rows])
    ax.plot(kn, e, color=col, lw=1.3, marker=mk, ms=3.2, mfc='white', mew=0.9)
    order = np.log(e[-2] / e[0]) / np.log(kn[-2] / kn[0])
    print(f'{key}: observed order {order:.2f} between Kn = {kn[0]} and {kn[-2]}')
clean_log_axes(ax)
ax.set_xticks([0.025, 0.05, 0.1, 0.2, 0.4]); ax.set_xticklabels(['0.025', '0.05', '0.1', '0.2', '0.4'])
ax.set_ylim(3e-7, 2.0); ax.set_yticks([1e-6, 1e-4, 1e-2, 1])
ax.set_xlabel('Knudsen number'); ax.set_ylabel(r'relative error of $Q^{-1}$', labelpad=1)
ax.text(0.024, 0.11, 'existing closure:\nsecond order', fontsize=6.0, ha='left', va='center', linespacing=1.05)
ax.text(0.42, 2.6e-6, 'this work:\nfourth order', fontsize=6.0, ha='right', va='center', linespacing=1.05)
fig.text(0.573, TITLE_Y, 'Phonon Boltzmann check', ha='center', fontsize=6.8, fontweight='bold')

# ------------------------------------------------------------------ right: silicon cantilever
l = 120e-9
a_um = np.logspace(np.log10(0.1), np.log10(5.0), 160)
q = {k: np.array([qinv_disk(a * 1e-6, l, k) for a in a_um]) for k in ('fourier', 'S', 'noslip', 'slip', 'perfect')}
ax = fig.add_axes([0.772, 0.215, 0.213, 0.53])
ax.fill_between(a_um, np.minimum(q['noslip'], q['perfect']), np.maximum(q['noslip'], q['perfect']), color=ORANGE, alpha=0.22, lw=0)
ax.plot(a_um, q['fourier'], color=GREY, lw=0.9, ls=(0, (4, 2)))
ax.plot(a_um, q['S'], color=BLUE, lw=1.4)
ax.plot(a_um, q['slip'], color=ORANGE, lw=1.4)
clean_log_axes(ax)
ax.set_xlim(0.1, 5.0); ax.set_ylim(4e-6, 4e-4)
ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5]); ax.set_xticklabels(['0.1', '0.2', '0.5', '1', '2', '5'])
ax.set_xlabel(r'beam radius ($\mu$m)'); ax.set_ylabel(r'$Q^{-1}$', labelpad=1)
i = int(np.argmin(abs(a_um - 0.2)))
drop = 100 * (q['S'][i] - q['noslip'][i]) / q['noslip'][i]
print(f"a = {a_um[i]:.3f} um: scalar {q['S'][i]:.3e}, no-slip {q['noslip'][i]:.3e}, slip {q['slip'][i]:.3e}, "
      f"perfect {q['perfect'][i]:.3e}; scalar relative to no-slip {drop:.0f}%")
ax.text(0.108, 2.2e-4, 'this work\n(no-slip to perfect slip)', fontsize=6.0, va='center', linespacing=1.05)
ax.text(0.33, 8.5e-6, 'existing closure', fontsize=6.0, va='center')
ax.annotate('Fourier', xy=(0.205, 1.82e-5), xytext=(0.115, 3.6e-5), fontsize=6.0, color=INK2, va='center', ha='left',
            arrowprops=dict(arrowstyle='-', color=GREY, lw=0.5, shrinkA=1, shrinkB=1))
fig.text(0.878, TITLE_Y, 'Silicon nanobeam', ha='center', fontsize=6.8, fontweight='bold')

for ext, kw in (('pdf', {}), ('png', dict(dpi=600))):
    fig.savefig(os.path.join(FIG, 'graphical_abstract.' + ext), **kw)
plt.close(fig)
try:
    from PIL import Image
    im = Image.open(os.path.join(FIG, 'graphical_abstract.png')).convert('RGB')
    im.save(os.path.join(FIG, 'graphical_abstract.tiff'), compression='tiff_lzw', dpi=(600, 600))
    print('pixels (w x h):', im.size)
except ImportError:
    print('Pillow not installed: TIFF not written')
