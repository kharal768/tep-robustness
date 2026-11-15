"""Comparison of the centre LP with the program of Dai et al. (bounds and timing)."""
import os
import sys, time; sys.argv = [sys.argv[0]]
from bounds_lib import *
import networkx as nx
def dai_lp(Z0, y0, eps, tmax=3000):
    nc = np.bincount(y0, minlength=4)
    U, cnt = np.unique(np.c_[Z0, y0], axis=0, return_counts=True); Z = U[:, :-1]; y = U[:, -1].astype(int); n = len(y)
    P = cKDTree(Z).query_pairs(2 * eps, p=np.inf, output_type='ndarray')
    G = nx.Graph(); G.add_edges_from(map(tuple, P))
    # only components containing a cross-class edge matter
    rows, cols, vals, rhs = [], [], [], []; r = 0; nz = 0; ncl = 0; t0 = time.time()
    for Q in nx.find_cliques(G):
        labs = np.unique(y[Q])
        if len(labs) < 2: continue
        ncl += 1; zid = {k: n + nz + j for j, k in enumerate(labs)}; nz += len(labs)
        rows += [r] * len(labs); cols += list(zid.values()); vals += [1] * len(labs); rhs.append(1); r += 1
        for i in Q: rows += [r, r]; cols += [i, zid[y[i]]]; vals += [1, -1]; rhs.append(0); r += 1
        if time.time() - t0 > tmax: return dict(dai_macro=np.nan, cliques=ncl, note='timeout')
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, n + nz)); w = cnt / (4 * nc[y]); N = n + nz
    res = linprog(np.r_[-w, np.zeros(nz)], A_ub=A, b_ub=np.array(rhs, float), bounds=(0, 1), method='highs')
    C = sparse.csr_matrix((-cnt / nc[y], (y, np.arange(n))), shape=(4, N))
    A2 = sparse.vstack([sparse.hstack([A, sparse.csr_matrix((A.shape[0], 1))]), sparse.hstack([C, sparse.csr_matrix(np.ones((4, 1)))])]).tocsr()
    r2 = linprog(np.r_[np.zeros(N), -1.0], A_ub=A2, b_ub=np.r_[np.array(rhs, float), np.zeros(4)], bounds=(0, 1), method='highs')
    return dict(dai_macro=-res.fun, dai_worst=-r2.fun, cliques=ncl)
if __name__ == '__main__':
    (Xtr, ytr, _), (Xv, yv, _), (Xh, yh, _) = load_sets(); rows = []
    for setn, Z, yy in (('heldout', Xh, yh), ('validation', Xv, yv)):
        for eps in (0.02, 0.05):
            t0 = time.time(); r1 = solve(Z, yy, eps, centres=True); m, wc = r1['macro'], r1['worst']; d = dai_lp(Z, yy, eps, tmax=2400)
            row = dict(set=setn, eps=eps, centre_macro=m, centre_worst=wc, **d, sec=time.time() - t0); rows.append(row)
            pd.DataFrame(rows).to_csv('results_r11_dai.csv', index=False); print(row, flush=True)
