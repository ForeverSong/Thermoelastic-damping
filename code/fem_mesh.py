"""Simple triangular meshes (no external mesher): graded disk and graded rectangle.
Returns dict(p=points (N,2), t=triangles (M,3) counter-clockwise, curved=bool)."""
import numpy as np
from scipy.spatial import Delaunay


def _ccw(p, t):
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    t = t.copy()
    neg = det < 0
    t[neg, 1], t[neg, 2] = t[neg, 2].copy(), t[neg, 1].copy()
    return t


def disk_mesh(h, grade=None, R=1.0):
    """Unit disk. h = target tangential/bulk spacing. grade = (thickness, h_wall): radial
    rings refined towards r = R inside a layer of given thickness (for Knudsen layers)."""
    radii = list(np.arange(R, 0, -h))
    if grade is not None:
        thick, hw = grade
        r, layer = R, []
        step = hw
        while r > R - thick:
            layer.append(r)
            r -= step
            step = min(step * 1.3, h)
        radii = sorted(set([round(x, 12) for x in layer + [x for x in radii if x < R - thick]]), reverse=True)
    pts = [np.zeros(2)]
    for r in radii:
        n = max(6, int(np.ceil(2 * np.pi * r / h)))
        if r == R:
            nR = n
        phi = 2 * np.pi * np.arange(n) / n + (0.0 if r == R else 0.5 * np.pi / n)
        pts.append(np.c_[r * np.cos(phi), r * np.sin(phi)])
    p = np.vstack([pts[0][None, :]] + pts[1:])
    tri = Delaunay(p)
    t = _ccw(p, tri.simplices)
    # drop slivers on the hull (Delaunay can create flat triangles on a convex polygon)
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    area = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    t = t[area > 1e-14 * h * h]
    return dict(p=p, t=t, curved=True, R=R)


def graded_1d(n, length, wall_h=None, ratio=1.25):
    """Nodes on [-length/2, length/2], geometric grading towards both ends if wall_h given."""
    if wall_h is None:
        return np.linspace(-length / 2, length / 2, n + 1)
    half, steps, s = length / 2, [], wall_h
    target = length / n
    pos = 0.0
    while pos + s < half - 1e-12:
        steps.append(s)
        pos += s
        s = min(s * ratio, target)
    steps[-1] += half - pos if pos < half else 0.0
    left = -half + np.concatenate([[0.0], np.cumsum(steps)])
    left[-1] = 0.0
    x = np.concatenate([left, -left[-2::-1]])
    return np.unique(np.round(x, 14))


def rect_mesh(b, h, nx, ny, wall_h=None):
    """Rectangle [-b/2,b/2] x [-h/2,h/2]; criss-cross-free right-triangle split."""
    xs = graded_1d(nx, b, wall_h)
    ys = graded_1d(ny, h, wall_h)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    p = np.c_[X.ravel(), Y.ravel()]
    NX, NY = len(xs), len(ys)
    idx = lambda i, j: i * NY + j
    t = []
    for i in range(NX - 1):
        for j in range(NY - 1):
            a, b_, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            if (i + j) % 2 == 0:
                t += [[a, b_, c], [a, c, d]]
            else:
                t += [[a, b_, d], [b_, c, d]]
    t = _ccw(p, np.array(t))
    return dict(p=p, t=t, curved=False, b=b, h=h)


def edges(t):
    """Unique edges, element-to-edge map (local edge k opposite local vertex k), boundary flags."""
    loc = np.array([[1, 2], [2, 0], [0, 1]])
    e_all = np.sort(t[:, loc].reshape(-1, 2), axis=1)
    e, inv, cnt = np.unique(e_all, axis=0, return_inverse=True, return_counts=True)
    t2e = inv.reshape(-1, 3)
    bnd = cnt == 1
    return e, t2e, bnd
