"""Library routines for the LP and integer feasibility bounds."""
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
