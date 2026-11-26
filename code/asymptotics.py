"""Check of the small-l asymptotics for the disk:
   (I_cons + 1) = (I_S + 1) * 1/(1 + 3 Lam L^2) * [1 - delta (1 - 1/(m rho1(m))) + O(L^2)],
   delta = L / (beta (1 + Cs beta)),  beta = sqrt(1 + i omega tau),  ls = Cs * l.
"""
import numpy as np
import gkted as g
Tt = 1j * 1e-3
beta = np.sqrt(1 + Tt)
for Om in (1.0, 3.39, 30.0):
    Lam = 1j * Om
    print(f"\nOmega = {Om}")
    print(f"{'L':>8} {'Cs':>5} {'exact wall corr':>28} {'1st-order':>28} {'ratio-1':>9}")
    for Cs in (0.0, 1.0, np.inf):
        for L in (0.1, 0.03, 0.01, 0.003):
            S = np.inf if np.isinf(Cs) else Cs * L
            Is = g.disk_I(Lam, L, Tt, S, 'slip')
            Ip = g.disk_I(Lam, L, Tt, S, 'P')
            m = np.sqrt(Lam * (1 + Tt) / (1 + 3 * Lam * L ** 2))
            exact = (Is + 1) / (Ip + 1) - 1
            delta = 0.0 if np.isinf(Cs) else L / (beta * (1 + Cs * beta))
            first = -delta * (1 - 1 / (m * g.rho1(m)))
            rr = abs(exact / first - 1) if first != 0 else np.nan
            print(f"{L:8.3f} {Cs:5.1f} {exact:28.3e} {first:28.3e} {rr:9.2e}")
# exactness of the normal-flux factor: (I_P+1)/(I_S+1) == 1/(1+3 Lam L^2)
Lam, L = 1j * 7.0, 0.2
Ip, Iss = g.disk_I(Lam, L, Tt, 0, 'P'), g.disk_I(Lam, L, Tt, 0, 'S')
print("\nnormal-flux factor check:", (Ip + 1) / (Iss + 1), 1 / (1 + 3 * Lam * L ** 2))
