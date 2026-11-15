"""52-variable rerun at the published budgets."""
import os
import glob, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bounds_centre_lp import solve
def load(prefix):
    X, Y, R, T = [], [], [], []
    for f in sorted(glob.glob(f'full/{prefix}_label*_seed*.dat')):
        c = int(f.split('label')[1][0]); d = np.loadtxt(f, skiprows=1)[:, 1:53]; t = np.arange(len(d))
        y = np.where((c > 0) & (t >= 40), c, 0)
        keep = ~((c > 0) & (t >= 40) & (t < 60))
        X.append(d[keep]); Y.append(y[keep]); R += [f] * keep.sum(); T.append(t[keep])
    return np.vstack(X), np.concatenate(Y), np.array(R), np.concatenate(T)
Xt, yt, rt, tt = load('tr'); Xh, yh, rh, th = load('ho')
mu, sd = Xt.mean(0), Xt.std(0)
# short-term variability from fault-free training runs
diffs = []
for r in np.unique(rt):
    if 'label0' not in r: continue
    Z = Xt[rt == r]; diffs.append(Z)
sig = np.zeros(52)
for j in range(52):
    dd = []
    for Z in diffs:
        v = Z[:, j]; u = v[np.r_[True, np.diff(v) != 0]]; dd.append(np.diff(u))
    sig[j] = np.concatenate(dd).std() / np.sqrt(2)
rows = []
for e in (0.1, 0.3):
    ratio = e * sd / sig
    rows.append(dict(kind='calib', eps=e, median_ratio=np.median(ratio), min_ratio=ratio.min(), max_ratio=ratio.max(),
                     analyzer_median=np.median(ratio[22:41]), analyzer_min=ratio[22:41].min(), analyzer_max=ratio[22:41].max(),
                     n_above_10=int((ratio > 10).sum())))
    print(rows[-1], flush=True)
Zh = (Xh - mu) / sd
for e in (0.1, 0.3):
    r0 = solve(Zh, yh, e, centres=False); r1 = solve(Zh, yh, e, centres=True)
    rows.append(dict(kind='bound', eps=e, n=len(yh), pair_macro=r0['macro'], centre_macro=r1['macro'], centre_worst=r1['worst'], edges=r1['edges']))
    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv('results_r17_fullvar.csv', index=False)
pd.DataFrame(dict(var=np.arange(1, 53), sd=sd, variability=sig, ratio01=0.1 * sd / sig, ratio03=0.3 * sd / sig)).to_csv('results_r17_fullvar_calib.csv', index=False)
