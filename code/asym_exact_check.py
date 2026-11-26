"""Check of the exact Knudsen-layer factor on the disk and of its expansion (Appendix A).
   With P = m rho(m), R = kappa rho(kappa), kappa = beta/L, S = ls/a:
     (I + 1)/(I_P + 1) = 1 - (P - 1)(1 - 2S) / [P R - 1 + S (P kappa^2 + 2 - 2 P R)]              (exact)
     ls = Cs l:  = 1 - delta (1 - 1/P) [1 + (L/beta)(1/2 + 1/P - 2 c^2)/(1 + c)] + O(L^3),  c = Cs beta
     perfect slip:  = 1 + 2 (1 - 1/P) L^2/beta^2 + O(L^3)
   Writes ../asym_exact_check.json"""
import numpy as np, json
import gkted as g

out = {}


def parts(Lam, L, Tt, alpha):
    m = np.sqrt(Lam * (1 + Tt) / (1 + (1 + alpha) * Lam * L ** 2)); beta = np.sqrt(1 + Tt); kap = beta / L
    return m * g.rho1(m), kap * g.rho1(kap), kap, beta


def exact_ratio(Lam, L, Tt, S, alpha):
    P, R, kap, beta = parts(Lam, L, Tt, alpha)
    if np.isinf(S):
        return 1 + 2 * (P - 1) / (P * kap ** 2 + 2 - 2 * P * R)
    return 1 - (P - 1) * (1 - 2 * S) / (P * R - 1 + S * (P * kap ** 2 + 2 - 2 * P * R))


# 1. the exact formula against the solution of the 2x2 boundary system
rng = np.random.default_rng(1); worst = 0.0; n = 0
for _ in range(20000):
    Om = 10 ** rng.uniform(-2, 4); L = 10 ** rng.uniform(-3, 0); wt = 10 ** rng.uniform(-4, 0); alpha = rng.choice([0.0, 1 / 3, 2.0])
    S = rng.choice([0.0, 10 ** rng.uniform(-3, 1), np.inf]); Lam, Tt = 1j * Om, 1j * wt
    ref = (g.disk_I(Lam, L, Tt, S, 'slip', alpha=alpha) + 1) / (g.disk_I(Lam, L, Tt, S, 'P', alpha=alpha) + 1)
    worst = max(worst, abs(exact_ratio(Lam, L, Tt, S, alpha) - ref) / abs(ref)); n += 1
print(f"exact factor vs 2x2 solve: {n} random cases, max relative difference {worst:.2e}")
out['exact_vs_solve'] = dict(cases=n, max_rel_diff=float(worst))

# 2. orders of the expansions
Ls = np.array([0.08, 0.04, 0.02, 0.01, 0.005]); rows = []
print(f"\n{'Omega':>6} {'wt':>5} {'alpha':>5} {'Cs':>4} | first-order residual, order | second-order residual, order")
for Om, wt, alpha in ((3.39, 1e-3, 2.0), (30.0, 1e-3, 2.0), (3.39, 0.3, 2.0), (3.39, 1e-3, 1 / 3), (300.0, 0.1, 1 / 3)):
    for Cs in (0.0, 0.5, 1.0, 3.0, np.inf):
        r1, r2 = [], []
        for L in Ls:
            Lam, Tt = 1j * Om, 1j * wt; P, R, kap, beta = parts(Lam, L, Tt, alpha); c = Cs * beta
            S = np.inf if np.isinf(Cs) else Cs * L; ex = exact_ratio(Lam, L, Tt, S, alpha)
            if np.isinf(Cs):
                r1.append(abs(ex - 1)); r2.append(abs(ex - (1 + 2 * (1 - 1 / P) * L ** 2 / beta ** 2)))
            else:
                d = L / (beta * (1 + c)); f1 = 1 - d * (1 - 1 / P)
                f2 = 1 - d * (1 - 1 / P) * (1 + (L / beta) * (0.5 + 1 / P - 2 * c ** 2) / (1 + c))
                r1.append(abs(ex - f1)); r2.append(abs(ex - f2))
        o1 = np.log2(r1[-2] / r1[-1]); o2 = np.log2(r2[-2] / r2[-1])
        print(f"{Om:6.2f} {wt:5.3f} {alpha:5.2f} {Cs:4.1f} | {r1[0]:.2e} -> {r1[-1]:.2e}, {o1:4.2f} | {r2[0]:.2e} -> {r2[-1]:.2e}, {o2:4.2f}")
        rows.append(dict(Omega=Om, wt=wt, alpha=alpha, Cs=(None if np.isinf(Cs) else Cs), res1=[float(x) for x in r1], res2=[float(x) for x in r2], order1=float(o1), order2=float(o2)))
out['L'] = Ls.tolist(); out['orders'] = rows

# 3. lower bound of |P| = |m rho(m)| over the parameter range (the expansion needs 1/P bounded)
Om = 10 ** np.linspace(-3, 6, 721)[:, None, None]; Lg = 10 ** np.linspace(-4, 0, 81)[None, :, None]; wt = np.array([0.0, 0.01, 0.1, 1.0, 10.0])[None, None, :]
for alpha in (0.0, 1 / 3, 2.0):
    m = np.sqrt(1j * Om * (1 + 1j * wt) / (1 + (1 + alpha) * 1j * Om * Lg ** 2)); Pm = np.abs(m * g.rho1(m))
    k = np.unravel_index(np.argmin(Pm), Pm.shape)
    print(f"alpha={alpha:.2f}: min |m rho(m)| = {Pm.min():.4f} at Omega={Om[k[0], 0, 0]:.3g}, L={Lg[0, k[1], 0]:.3g}, wt={wt[0, 0, k[2]]:.3g}")
    out[f'minP_alpha_{alpha:.2f}'] = float(Pm.min())
json.dump(out, open('../asym_exact_check.json', 'w'), indent=1)
