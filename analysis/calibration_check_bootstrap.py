"""Calibration check and two-way bootstrap confidence intervals with Holm correction."""
import os
import sys, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy import stats
from calibration_bounds import load_all, inv, BL
X, y, rid, t, itr, iv = load_all()
nom = np.array([str(r).startswith('label0') for r in rid]) & np.isin(np.arange(len(y)), itr)
zbar = X[nom].mean(0); cal = pd.read_csv('results_r5_calib.csv'); cal = cal[cal.kind == 'part']; rows = []
rng = np.random.default_rng(0)
for a, b, name, parts in BL:
    zb = zbar[a:b]; x0 = inv(zb); q = len(zb)
    for i, p in enumerate(parts):
        r = dict(block=name, part=p, share=x0[i])
        for eps in (0.02, 0.15):
            best = 0.0
            for sgn in (1, -1):
                f = lambda d: -sgn * (inv(zb + d)[i] - x0[i])
                for k in range(20):
                    d0 = rng.uniform(-eps, eps, q) if k else np.zeros(q)
                    res = minimize(f, d0, bounds=[(-eps, eps)] * q, method='L-BFGS-B')
                    best = max(best, -res.fun)
            corner = cal[(cal.block == name) & (cal.part == p)][f'max_change_{eps}'].values[0]
            r[f'opt_{eps}'] = best; r[f'corner_{eps}'] = corner
        rows.append(r)
C = pd.DataFrame(rows); C.to_csv('results_r7_calib_opt.csv', index=False)
C['rel02'] = C.opt_002 / C.corner_002 if 'opt_002' in C else None
print(C.round(5).to_string()); print('max rel diff', (C['opt_0.02'] / C['corner_0.02']).max(), (C['opt_0.15'] / C['corner_0.15']).max())

out = []
def per_run_seed(fname_fmt, cls, seeds):
    M = {}
    for s in seeds:
        z = np.load(fname_fmt.format(s=s), allow_pickle=True); ok, yy, rr = z['ok'], z['y'], z['rid']
        for r in np.unique(rr[yy == cls]): M.setdefault(r, {})[s] = ok[(yy == cls) & (rr == r)].mean()
    return M
specs = [(b, tg, f'persample/{b}_{{tag}}_{{s}}.npz') for b in ('raw002', 'sd025') for tg in ('adv_pgd4', 'adv_pgd10', 'tuned')] + [('cnn', 'adversarial', 'persample/cnn_{tag}_{s}.npz')]
seeds = list(range(5000, 5005))
for b, tg, fmt in specs:
    for cls in range(4):
        S = per_run_seed(fmt.replace('{tag}', 'standard'), cls, seeds); A = per_run_seed(fmt.replace('{tag}', tg), cls, seeds)
        runs_ = sorted(S); D = np.array([[A[r][s] - S[r][s] for s in seeds] for r in runs_])   # runs x seeds
        d = D.mean(1); n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n); tq = stats.t.ppf(.975, n - 1)
        # two-way (run, seed) bootstrap
        bs = []
        for _ in range(5000):
            ri = rng.integers(0, n, n); si = rng.integers(0, len(seeds), len(seeds)); bs.append(D[np.ix_(ri, si)].mean())
        out.append(dict(budget=b, model=tg, cls=cls, runs=n, diff=m, t_lo=m - tq * se, t_hi=m + tq * se, bs2_lo=np.quantile(bs, .025), bs2_hi=np.quantile(bs, .975)))
O = pd.DataFrame(out); O.to_csv('results_r7_ci.csv', index=False); print(O.round(3).to_string())
