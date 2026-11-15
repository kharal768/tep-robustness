"""Feasibility bounds at the search budget eps = 0.15."""
import os
import sys; sys.argv=[sys.argv[0]]
from bounds_centre_lp import *
def build_c(Z0, y0, eps, stride=1):
    nc = np.bincount(y0, minlength=4)
    U, cnt = np.unique(np.c_[Z0, y0], axis=0, return_counts=True); Z = U[:, :-1]; y = U[:, -1].astype(int); n = len(y)
    tree = cKDTree(Z); nb = tree.query_ball_point(Z[::stride], eps, p=np.inf)
    rows, cols, vals, rhs = [], [], [], []; r = 0; nz = 0
    for lst in nb:
        lst = np.asarray(lst); labs = np.unique(y[lst])
        if len(labs) < 2: continue
        zid = {k: n + nz + j for j, k in enumerate(labs)}; nz += len(labs)
        rows += [r] * len(labs); cols += list(zid.values()); vals += [1] * len(labs); rhs.append(1); r += 1
        m = len(lst); rr = np.arange(r, r + m)
        rows += list(rr) + list(rr); cols += list(lst) + [zid[k] for k in y[lst]]; vals += [1] * m + [-1] * m; rhs += [0] * m; r += m
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, n + nz)); return A, np.array(rhs, float), cnt / (4 * nc[y]), y, cnt, nc, n, nz
def solve_c(Z0, y0, eps, stride=1):
    A, b, w, y, cnt, nc, n, nz = build_c(Z0, y0, eps, stride); N = n + nz
    r = linprog(np.r_[-w, np.zeros(nz)], A_ub=A, b_ub=b, bounds=(0, 1), method='highs')
    C = sparse.csr_matrix((-cnt / nc[y], (y, np.arange(n))), shape=(4, N))
    A2 = sparse.vstack([sparse.hstack([A, sparse.csr_matrix((A.shape[0], 1))]), sparse.hstack([C, sparse.csr_matrix(np.ones((4, 1)))])]).tocsr()
    r2 = linprog(np.r_[np.zeros(N), -1.0], A_ub=A2, b_ub=np.r_[b, np.zeros(4)], bounds=(0, 1), method='highs')
    return -r.fun, -r2.fun, A.shape[0]
(Xtr, ytr, _), (Xv, yv, _), (Xh, yh, _) = load_sets(); sd = Xtr.std(0); rows = []
for setn, Z, yy in (('heldout', Xh, yh),):
    for coords, scl, eps in (('raw', 1.0, 0.02), ('sd', sd, 0.25), ('sd', sd, 0.1), ('raw', 1.0, 0.05)):
        A = Z / scl; r0 = solve(A, yy, eps, centres=False); r1 = solve(A, yy, eps, centres=True)
        row = dict(set=setn, coords=coords, eps=eps, lp_pair_macro=r0['macro'], lp_pair_worst=r0['worst'], lp_centre_macro=r1['macro'], lp_centre_worst=r1['worst'], edges=r1['edges'])
        if r1['edges'] < 200000: ri = solve(A, yy, eps, centres=True, integer=True, tl=1200); row.update(ip_macro=ri['macro'], ip_status=ri.get('status'))
        rows.append(row); print(row, flush=True); pd.DataFrame(rows).to_csv('results_r6_bounds_heldout.csv', index=False)
for setn, Z, yy in (('validation', Xv, yv), ('heldout', Xh, yh)):
    t0 = time.time(); mac, wor, nr = solve_c(Z, yy, 0.15)
    rows.append(dict(set=setn, coords='raw', eps=0.15, lp_centre_only_macro=mac, lp_centre_only_worst=wor, rows=nr, sec=time.time() - t0)); print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv('results_r6_bounds_heldout.csv', index=False)
