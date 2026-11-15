"""Centre-constrained LP and worst-class feasibility bounds."""
import os
import sys, json, types, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','core')); sys.modules['mobo'] = types.ModuleType('mobo'); sys.modules['mobo'].run_mobo = None
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy import sparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_sets import load_sets

def build(Z0, y0, eps, centres=True):
    nc = np.bincount(y0, minlength=4)
    U, cnt = np.unique(np.c_[Z0, y0], axis=0, return_counts=True); Z = U[:, :-1]; y = U[:, -1].astype(int); n = len(y)
    tree = cKDTree(Z); P = tree.query_pairs(2 * eps, p=np.inf, output_type='ndarray'); P = P[y[P[:, 0]] != y[P[:, 1]]]
    rows, cols, vals, rhs = [], [], [], []; r = 0
    for i, j in P: rows += [r, r]; cols += [i, j]; vals += [1, 1]; rhs.append(1); r += 1
    nz = 0
    if centres:
        nb = tree.query_ball_point(Z, eps, p=np.inf)
        zid = {}
        for c, lst in enumerate(nb):
            labs = set(y[lst])
            if len(labs) < 2: continue
            for k in labs:
                zid[(c, k)] = n + nz; nz += 1
            for k in labs: rows.append(r); cols.append(zid[(c, k)]); vals.append(1)
            rhs.append(1); r += 1
            for i in lst: rows += [r, r]; cols += [i, zid[(c, y[i])]]; vals += [1, -1]; rhs.append(0); r += 1
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, n + nz)); w = cnt / (4 * nc[y])
    return A, np.array(rhs, float), w, y, cnt, nc, n, nz, len(P)

def solve(Z0, y0, eps, centres=True, integer=False, tl=1800):
    A, b, w, y, cnt, nc, n, nz, ne = build(Z0, y0, eps, centres)
    N = n + nz; out = dict(edges=ne, centres_vars=nz)
    if A.shape[0] == 0: return dict(macro=1.0, worst=1.0, **out)
    c = np.r_[-w, np.zeros(nz)]
    if integer:
        r = milp(c, constraints=LinearConstraint(A, -np.inf, b), integrality=np.ones(N), bounds=Bounds(0, 1), options=dict(time_limit=tl, disp=False))
        out['macro'] = -r.fun if r.x is not None else np.nan; out['macro_dual'] = getattr(r, 'mip_dual_bound', np.nan); out['status'] = r.status
        out['macro_dual'] = -out['macro_dual'] if out['macro_dual'] is not None and not np.isnan(out['macro_dual']) else np.nan
    else:
        r = linprog(c, A_ub=A, b_ub=b, bounds=(0, 1), method='highs'); out['macro'] = -r.fun
        C = sparse.csr_matrix((-cnt / nc[y], (y, np.arange(n))), shape=(4, N))
        A2 = sparse.vstack([sparse.hstack([A, sparse.csr_matrix((A.shape[0], 1))]), sparse.hstack([C, sparse.csr_matrix(np.ones((4, 1)))])]).tocsr()
        r2 = linprog(np.r_[np.zeros(N), -1.0], A_ub=A2, b_ub=np.r_[b, np.zeros(4)], bounds=[(0, 1)] * N + [(0, 1)], method='highs'); out['worst'] = -r2.fun
    return out

def bounds():
    (Xtr, ytr, _), (Xv, yv, _), (Xh, yh, _) = load_sets(); sd = Xtr.std(0); rows = []
    for setn, Z, yy in (('validation', Xv, yv), ('heldout', Xh, yh)):
        for coords, scl, eps in (('raw', 1.0, 0.02), ('sd', sd, 0.25), ('sd', sd, 0.1), ('raw', 1.0, 0.05), ('raw', 1.0, 0.15)):
            A = Z / scl; t0 = time.time()
            r0 = solve(A, yy, eps, centres=False); r1 = solve(A, yy, eps, centres=True)
            row = dict(set=setn, coords=coords, eps=eps, n=len(yy), lp_pair_macro=r0['macro'], lp_pair_worst=r0['worst'], lp_centre_macro=r1['macro'], lp_centre_worst=r1['worst'], edges=r1['edges'])
            if r1['edges'] < 200000:
                ri = solve(A, yy, eps, centres=True, integer=True, tl=1200); row.update(ip_macro=ri['macro'], ip_dual=ri.get('macro_dual'), ip_status=ri.get('status'))
            row['sec'] = time.time() - t0; rows.append(row); pd.DataFrame(rows).to_csv('results_r6_bounds.csv', index=False); print(row, flush=True)

def overlap():
    (Xtr, ytr, _), (Xv, yv, _), _ = load_sets(); sd = Xtr.std(0); rows = []
    for coords, scl, grid in (('raw', 1.0, [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2]), ('sd', sd, [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0])):
        dist = np.full(len(yv), np.inf)
        for c in range(4):
            T = cKDTree(Xtr[ytr != c] / scl); m = yv == c; d, _ = T.query(Xv[m] / scl, p=np.inf); dist[m] = d
        for e in grid:
            r = dict(coords=coords, eps=e, all=(dist < e).mean(), median_dist=np.median(dist)); r.update({f'c{c}': (dist[yv == c] < e).mean() for c in range(4)}); rows.append(r)
    pd.DataFrame(rows).to_csv('results_r6_overlap.csv', index=False); print(pd.DataFrame(rows).round(4).to_string())

def boot():
    rng = np.random.default_rng(0); rows = []
    def per_run(budget, tag, cls, pref='', seeds=range(5000, 5005)):
        acc = {}
        for s in seeds:
            z = np.load(f'persample/{pref}{budget}_{tag}_{s}.npz' if budget else f'persample/{pref}{tag}_{s}.npz', allow_pickle=True)
            ok, y, rid = z['ok'], z['y'], z['rid']
            for r in np.unique(rid[y == cls]): acc.setdefault(r, []).append(ok[(y == cls) & (rid == r)].mean())
        return {k: np.mean(v) for k, v in acc.items()}
    specs = [(b, t, '') for b in ('raw002', 'sd025') for t in ('adv_pgd4', 'adv_pgd10', 'tuned')] + [(None, 'adversarial', 'cnn_')]
    for b, tag, pref in specs:
        for cls in range(4):
            S = per_run(b, 'standard', cls, pref); A = per_run(b, tag, cls, pref); ks = sorted(S); d = np.array([A[k] - S[k] for k in ks])
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
            rows.append(dict(budget=b or 'cnn_raw002', model=tag, cls=cls, runs=len(d), diff=d.mean(), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975)))
    R = pd.DataFrame(rows); R.to_csv('results_r6_boot.csv', index=False); print(R.round(3).to_string())

if __name__ == '__main__':
    {'bounds': bounds, 'overlap': overlap, 'boot': boot}[sys.argv[1]]()
