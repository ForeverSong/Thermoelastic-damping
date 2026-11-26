"""Field figures (set-up, disk/rectangle/ellipse fields) and the elliptical-section test.
Writes ../figures/fig_setup.*, fig_fields.*, ../tables/tab_ellipse.tex, ../ellipse_results.json"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from scipy.interpolate import griddata
import gkted as g, fem_mesh as fm, fem_mixed_curved as mc, fem_scalar as fs

HERE = os.path.dirname(os.path.abspath(__file__))
FIG, TAB = os.path.join(HERE, '..', 'figures'), os.path.join(HERE, '..', 'tables')
C1, C2, C3, C4 = '#257de0', '#f47230', '#00b894', '#f2a600'
GREY, INK, INK2 = '#8c8c8c', '#191919', '#666666'
plt.rcParams.update({'font.size': 8.5, 'axes.labelsize': 8.5, 'legend.fontsize': 7.5, 'axes.edgecolor': INK2,
                     'axes.linewidth': 0.6, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK,
                     'text.color': INK, 'savefig.dpi': 300})
SEQ = LinearSegmentedColormap.from_list('seq', ['#f4f8fd', '#b9d6f5', '#5a9de8', '#1f5fbf', '#0b2e6b'])
DIV = LinearSegmentedColormap.from_list('div', ['#0b3d91', '#257de0', '#a9c9f0', '#f2f2f0', '#f8b995', '#f47230', '#8a2f00'])
Tt = 1e-3j


def disk_xy(Lam, L, S, model, n=301):
    c = g.disk_coeffs(Lam, L, Tt, S, model); c['Lam'] = Lam
    y = np.linspace(-1, 1, n); Y, Z = np.meshgrid(y, y)
    R = np.hypot(Y, Z); PH = np.arctan2(Z, Y); out = R > 1.0
    r = np.clip(R, 1e-5, 1.0).ravel()
    Qr, Qp, dQr, dQp, Th = [np.asarray(a).reshape(R.shape) for a in g.disk_fields(c, r)]
    rr = r.reshape(R.shape)
    qr, qp = Qr * np.sin(PH), Qp * np.cos(PH)
    qy, qz = qr * np.cos(PH) - qp * np.sin(PH), qr * np.sin(PH) + qp * np.cos(PH)
    om = (dQp + Qp / rr - Qr / rr) * np.cos(PH)
    ma = lambda a: np.ma.array(np.imag(a), mask=out)
    wall = g.disk_fields(c, np.array([1.0]))[0][0]          # Q_r(1): normal flux amplitude at the wall
    return Y, Z, ma(qy), ma(qz), ma(om), wall


def fe_fields(S, sol):
    xc, qv, om = [], [], []
    for el in S['els']:
        x, D, val, grad, J = mc._phys(el['Xe'], np.array([1 / 3, 1 / 3]), el['sg'])
        qd = sol['q'][el['gd']]
        xc.append(x); qv.append(qd @ val); om.append(qd @ (grad[:, 1, 0] - grad[:, 0, 1]))
    return np.array(xc), np.array(qv), np.array(om)


def p2_edges(S, only_boundary=False):
    """polylines of all (curved) element edges"""
    s = np.linspace(0, 1, 9); N = np.array([(1 - s) * (1 - 2 * s), 4 * s * (1 - s), s * (2 * s - 1)]).T
    loc = [(1, 2, 3), (2, 0, 4), (0, 1, 5)]; segs = []
    for k, el in enumerate(S['els']):
        for j, (a, b, c) in enumerate(loc):
            if only_boundary and not S['bnd'][S['t2e'][k, j]]: continue
            segs.append(N @ el['Xe'][[a, c, b]])
    return segs


def stream(ax, xc, qv, inside, xlim, ylim, n=160, density=1.1):
    X, Y = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    U = griddata(xc, np.imag(qv[:, 0]), (X, Y), method='linear'); V = griddata(xc, np.imag(qv[:, 1]), (X, Y), method='linear')
    m = ~inside(X, Y) | np.isnan(U)
    ax.streamplot(X, Y, np.ma.array(U, mask=m), np.ma.array(V, mask=m), density=density, color=INK, linewidth=0.45, arrowsize=0.5)


# ------------------------------------------------------------------ elliptical sections (no closed form)
OUT = os.path.join(HERE, '..', 'ellipse_results.json')
Lam, L = 3.39j, 0.1
if os.path.exists(OUT):
    E = json.load(open(OUT))
else:
    E = dict(rows=[])
    for ar in (0.5, 1.0, 2.0, 4.0):
        row = dict(ar=ar)
        for h in (0.1, 0.05):
            mesh = fm.disk_mesh(h, grade=(0.4, (L / 4) * (h / 0.2))); mesh['scale'] = (ar, 1.0)
            S = mc.build(mesh); A = mc.assemble(S, L)
            for key, s in (('noslip', 0.0), ('slip', L), ('perfect', np.inf)):
                r = mc.solve(A, Lam, L, Tt, s)
                row[f'{key}_{h}'] = float(-np.imag(r['I']))
                row[f'{key}_id_{h}'] = float(abs(abs(np.imag(r['I'])) - r['D'] / (r['Omega'] * A['zz'])) / abs(np.imag(r['I'])))
        ms = fm.disk_mesh(0.05); ms['scale'] = (ar, 1.0); As = fs.assemble(ms)
        row['scalar'] = float(-np.imag(fs.solve(As, Lam, L, Tt, 'S')[0])); row['fourier'] = float(-np.imag(fs.solve(As, Lam, 0.0, 0.0, 'S')[0]))
        row['area_err'] = float(abs(As['area'] - np.pi * ar) / (np.pi * ar))
        E['rows'].append(row); print(row, flush=True)
    E['disk_exact'] = {k: float(-np.imag(g.disk_I(Lam, L, Tt, s, 'slip'))) for k, s in (('noslip', 0.0), ('slip', L), ('perfect', np.inf))}
    json.dump(E, open(OUT, 'w'), indent=1)
with open(os.path.join(TAB, 'tab_ellipse.tex'), 'w') as f:
    for r in E['rows']:
        d = max(abs(r[f'{k}_0.05'] - r[f'{k}_0.1']) / r[f'{k}_0.05'] for k in ('noslip', 'slip', 'perfect'))
        f.write(f"{r['ar']:g} & {r['fourier']:.4f} & {r['scalar']:.4f} & {r['noslip_0.05']:.4f} & {r['slip_0.05']:.4f} & {r['perfect_0.05']:.4f} & "
                f"{100*(r['noslip_0.05']-r['scalar'])/r['scalar']:+.1f} & {100*(r['perfect_0.05']-r['noslip_0.05'])/r['noslip_0.05']:+.1f} & {100*d:.2f} \\\\\n")
print(open(os.path.join(TAB, 'tab_ellipse.tex')).read(), E['disk_exact'])

# ------------------------------------------------------------------ figure: fields
fig = plt.figure(figsize=(6.9, 4.9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.72], hspace=0.22, wspace=0.12, left=0.02, right=0.86, top=0.95, bottom=0.04)
th = np.linspace(0, 2 * np.pi, 400)
for j, (model, S_, title) in enumerate((('S', 0.0, r'(a) scalar closure: $\mathrm{Im}\,\mathbf{q}$'), ('slip', 0.0, r'(b) consistent, no-slip: $\mathrm{Im}\,\mathbf{q}$'))):
    ax = fig.add_subplot(gs[0, j]); Y, Z, qy, qz, om, wall = disk_xy(Lam, L, S_, model)
    mag = np.ma.sqrt(qy ** 2 + qz ** 2); mq = mag.max() if j == 0 else mq
    im1 = ax.pcolormesh(Y, Z, mag / mq, cmap=SEQ, vmin=0, vmax=1, shading='auto', rasterized=True)
    ax.streamplot(Y, Z, qy, qz, density=1.0, color=INK, linewidth=0.45, arrowsize=0.5)
    ax.plot(np.cos(th), np.sin(th), color=INK, lw=1.0)
    if model == 'S':
        a = np.linspace(0, 2 * np.pi, 28, endpoint=False); qn = np.imag(wall) * np.sin(a)
        ax.quiver(np.cos(a), np.sin(a), qn * np.cos(a), qn * np.sin(a), color=C2, scale=1.1, width=0.014, zorder=5)
        ax.text(0, -1.27, 'flux through the wall', color=C2, ha='center', fontsize=7.5)
    ax.set_title(title, fontsize=8.5, loc='left'); ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-1.32, 1.32); ax.set_ylim(-1.34, 1.2)
ax = fig.add_subplot(gs[0, 2]); Y, Z, qy, qz, om, wall = disk_xy(Lam, L, 0.0, 'slip', n=601)
im2 = ax.pcolormesh(Y, Z, om / np.abs(om).max(), cmap=DIV, vmin=-1, vmax=1, shading='auto', rasterized=True)
ax.plot(np.cos(th), np.sin(th), color=INK, lw=1.0)
ax.set_title(r'(c) no-slip: vorticity $\nabla\times\mathbf{q}$', fontsize=8.5, loc='left'); ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-1.32, 1.32); ax.set_ylim(-1.34, 1.2)
# (d) rectangle, mixed FE
B = 2; Lr, Lamr = 0.1, 1j * np.pi ** 2
Sr = mc.build(fm.rect_mesh(B, 1.0, 20 * B, 20, wall_h=Lr / 8)); Ar = mc.assemble(Sr, Lr); rr = mc.solve(Ar, Lamr, Lr, Tt, 0.0)
xc, qv, om = fe_fields(Sr, rr); ax = fig.add_subplot(gs[1, 0])
ax.tripcolor(Sr['p'][:, 0], Sr['p'][:, 1], Sr['t'], facecolors=np.imag(om) / np.percentile(np.abs(np.imag(om)), 99.5), cmap=DIV, vmin=-1, vmax=1, rasterized=True)
stream(ax, xc, qv, lambda X, Y: (abs(X) < B / 2) & (abs(Y) < 0.5), (-B / 2, B / 2), (-0.5, 0.5), density=0.9)
ax.plot([-1, 1, 1, -1, -1], [-.5, -.5, .5, .5, -.5], color=INK, lw=1.0)
ax.set_title(r'(d) rectangle $b=2h$: vorticity, FE', fontsize=8.5, loc='left'); ax.set_aspect('equal'); ax.axis('off')
# (e,f) ellipse, mixed FE
ar = 2.0; mesh = fm.disk_mesh(0.08, grade=(0.4, L / 10)); mesh['scale'] = (ar, 1.0)
Se = mc.build(mesh); Ae = mc.assemble(Se, L); re_ = mc.solve(Ae, Lam, L, Tt, 0.0); xc, qv, om = fe_fields(Se, re_)
vm = np.percentile(np.abs(np.imag(om)), 99.5)
ax = fig.add_subplot(gs[1, 1])
ax.tripcolor(Se['p'][:, 0], Se['p'][:, 1], Se['t'], facecolors=np.imag(om) / vm, cmap=DIV, vmin=-1, vmax=1, rasterized=True)
stream(ax, xc, qv, lambda X, Y: (X / ar) ** 2 + Y ** 2 < 0.985, (-ar, ar), (-1, 1), density=0.9)
ax.plot(ar * np.cos(th), np.sin(th), color=INK, lw=1.0)
ax.add_patch(plt.Rectangle((1.3, -0.3), 0.75, 0.6, fill=False, ec=INK, lw=0.7, ls='--'))
ax.set_title(r'(e) ellipse $2{:}1$: vorticity, FE', fontsize=8.5, loc='left'); ax.set_aspect('equal'); ax.axis('off')
ax = fig.add_subplot(gs[1, 2])
ax.tripcolor(Se['p'][:, 0], Se['p'][:, 1], Se['t'], facecolors=np.imag(om) / vm, cmap=DIV, vmin=-1, vmax=1, rasterized=True)
ax.add_collection(LineCollection(p2_edges(Se), colors='#33333366', linewidths=0.3))
ax.add_collection(LineCollection(p2_edges(Se, True), colors=INK, linewidths=1.0))
ax.set_xlim(1.3, 2.05); ax.set_ylim(-0.3, 0.3); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
ax.set_title('(f) zoom: curved mesh', fontsize=8.5, loc='left')
cax1 = fig.add_axes([0.885, 0.56, 0.014, 0.3]); cb = fig.colorbar(im1, cax=cax1); cb.set_label(r'$|\mathrm{Im}\,\mathbf{q}|$ (normalised)', fontsize=7.5); cb.ax.tick_params(labelsize=7)
cax2 = fig.add_axes([0.885, 0.1, 0.014, 0.3]); cb = fig.colorbar(im2, cax=cax2); cb.set_label(r'$\mathrm{Im}\,\omega$ (normalised)', fontsize=7.5); cb.ax.tick_params(labelsize=7)
fig.savefig(os.path.join(FIG, 'fig_fields.pdf')); fig.savefig(os.path.join(FIG, 'fig_fields.png'), dpi=160); plt.close(fig)

# ------------------------------------------------------------------ figure: set-up and mesh
fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.45), gridspec_kw=dict(wspace=0.08, left=0.02, right=0.98, top=0.9, bottom=0.03))
ax = axs[0]; n = 301; y = np.linspace(-1, 1, n); Y, Z = np.meshgrid(y, y); R = np.hypot(Y, Z)
ax.pcolormesh(Y, Z, np.ma.array(-Z, mask=R > 1), cmap=DIV, vmin=-1.6, vmax=1.6, shading='auto', rasterized=True)
ax.fill(np.r_[np.cos(th), 0.86 * np.cos(th[::-1])], np.r_[np.sin(th), 0.86 * np.sin(th[::-1])], color='white', alpha=0.45, lw=0)
ax.plot(np.cos(th), np.sin(th), color=INK, lw=1.1); ax.plot(0.86 * np.cos(th), 0.86 * np.sin(th), color=INK, lw=0.5, ls=(0, (3, 2)))
for x0 in (-0.45, 0.0, 0.45):
    ax.annotate('', xy=(x0, 0.5), xytext=(x0, -0.5), arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.9))
ax.text(0.06, 0.02, r'$\mathbf{q}$', fontsize=9); ax.text(-1.55, 1.12, r'colour: adiabatic temperature $-z$', fontsize=7.2)
ax.annotate(r'$\mathbf{q}\cdot\mathbf{n}=0$' + '\n' + r'$\mathbf{q}\cdot\mathbf{t}=-\ell_s\,\mathbf{t}\cdot2\varepsilon(\mathbf{q})\mathbf{n}$', xy=(0.87, -0.5), xytext=(0.12, -1.5), fontsize=7.2,
            arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6), ha='left')
ax.annotate(r'Knudsen layer, $\delta\sim l$', xy=(-0.93 * np.cos(0.5), 0.93 * np.sin(0.5)), xytext=(-1.6, -1.2), fontsize=7.2, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6))
ax.annotate('', xy=(1.22, 0.45), xytext=(1.22, -0.05), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.7)); ax.text(1.25, 0.2, '$z$', fontsize=8)
ax.set_title('(a) cross-section problem', fontsize=8.5, loc='left'); ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-1.65, 1.6); ax.set_ylim(-1.75, 1.3)
Sm = mc.build(fm.disk_mesh(0.2, grade=(0.4, 0.025)))
ax = axs[1]; ax.add_collection(LineCollection(p2_edges(Sm), colors=INK2, linewidths=0.35)); ax.add_collection(LineCollection(p2_edges(Sm, True), colors=INK, linewidths=1.0))
ax.add_patch(plt.Rectangle((0.62, 0.5), 0.36, 0.36, fill=False, ec=C2, lw=0.9))
ax.set_title('(b) graded mesh, curved wall', fontsize=8.5, loc='left'); ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08)
ax = axs[2]; ax.add_collection(LineCollection(p2_edges(Sm), colors=INK2, linewidths=0.5)); ax.add_collection(LineCollection(p2_edges(Sm, True), colors=INK, linewidths=1.4))
chords = [np.array([el['Xe'][a], el['Xe'][b]]) for k, el in enumerate(Sm['els']) for j, (a, b) in enumerate([(1, 2), (2, 0), (0, 1)]) if Sm['bnd'][Sm['t2e'][k, j]]]
ax.add_collection(LineCollection(chords, colors=C2, linewidths=1.0, linestyles='--'))
ax.plot([], [], color=INK, lw=1.4, label='$P_2$ wall (Piola-mapped BDM$_1$)'); ax.plot([], [], color=C2, lw=1.0, ls='--', label='chords (straight-sided)')
ax.legend(frameon=False, fontsize=6.8, loc='lower left', handlelength=1.8)
ax.set_xlim(0.62, 0.98); ax.set_ylim(0.5, 0.86); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_edgecolor(C2); sp.set_linewidth(0.9)
ax.set_title('(c) zoom', fontsize=8.5, loc='left')
fig.savefig(os.path.join(FIG, 'fig_setup.pdf')); fig.savefig(os.path.join(FIG, 'fig_setup.png'), dpi=160)
