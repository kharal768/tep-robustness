"""Analyzer-calibrated budgets (P1) and feasibility bounds (P2) for the main fault classes."""
import os
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','core')); import types; sys.modules['mobo']=types.ModuleType('mobo'); sys.modules['mobo'].run_mobo=None
from run_pipeline import load_csv_dataset, split
from coda import _helmert_ilr_basis
from scipy.spatial import cKDTree
from scipy.optimize import linprog
from scipy import sparse
D = os.environ.get('TEP_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data'))+'/'
BL = [(0, 5, 'reactor feed', list('ABCDEF')), (5, 12, 'purge', list('ABCDEFGH')), (12, 16, 'product', list('DEFGH'))]

def inv(zb):
    V = _helmert_ilr_basis(zb.shape[-1] + 1); clr = zb @ V; e = np.exp(clr); return e / e.sum(-1, keepdims=True)

def load_all():
    X, y, rid, t, _ = load_csv_dataset(D + 'raw_dataset/tep_multirun_composition_data.csv')
    idx = np.arange(len(y), dtype=float)[:, None]
    itr, _, iv, _ = split(idx, y, run_id=rid, t=t, val_frac=0.25, seed=0)
    return X, y, rid, t, itr[:, 0].astype(int), iv[:, 0].astype(int)

def calib():
    X, y, rid, t, itr, iv = load_all()
    nom = np.array([str(r).startswith('label0') for r in rid]) & np.isin(np.arange(len(y)), itr)
    rows = []
    # analyzer noise: successive differences of distinct consecutive values within normal-operation runs
    dz = []
    for r in np.unique(rid[nom]):
        Z = X[(rid == r)][np.argsort(t[rid == r])]
        keep = np.r_[True, np.any(np.diff(Z, axis=0) != 0, axis=1)]; Zd = Z[keep]
        dz.append(np.diff(Zd, axis=0))
    dz = np.vstack(dz); noise = dz.std(0) / np.sqrt(2)
    sd_nom = X[nom].std(0)
    for j in range(16):
        rows.append(dict(kind='coord', coord=j + 1, noise_sd=noise[j], nominal_sd=sd_nom[j], eps002_over_noise=0.02 / noise[j], eps015_over_noise=0.15 / noise[j]))
    # translate eps into changes of part shares at the mean normal composition (exact evaluation at the gradient-sign corner)
    zbar = X[nom].mean(0)
    for a, b, name, parts in BL:
        zb = zbar[a:b]; x0 = inv(zb)
        # part noise in shares
        xs = []
        for r in np.unique(rid[nom]):
            Z = X[(rid == r)][np.argsort(t[rid == r])][:, a:b]
            keep = np.r_[True, np.any(np.diff(Z, axis=0) != 0, axis=1)]; xs.append(np.diff(inv(Z[keep]), axis=0))
        xn = np.vstack(xs).std(0) / np.sqrt(2)
        for i, p in enumerate(parts):
            out = dict(kind='part', block=name, part=p, share=x0[i], noise_share=xn[i])
            for eps in (0.02, 0.05, 0.15):
                best = 0.0
                for sgn in (1, -1):
                    # numerical gradient of share i
                    g = np.array([(inv(zb + 1e-6 * np.eye(len(zb))[k])[i] - x0[i]) / 1e-6 for k in range(len(zb))])
                    d = sgn * eps * np.sign(g); best = max(best, abs(inv(zb + d)[i] - x0[i]))
                out[f'max_change_{eps}'] = best; out[f'max_change_over_noise_{eps}'] = best / xn[i]
            rows.append(out)
    pd.DataFrame(rows).to_csv('results_r5_calib.csv', index=False)
    print(pd.DataFrame(rows).round(5).to_string())

def lp_bounds(Z0, y0, eps, label):
    # identical rows with the same label have identical balls, so they share robust correctness: merge them (exact)
    nc = np.bincount(y0, minlength=4)
    key = np.c_[Z0, y0]; U, inv_, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    Z = U[:, :-1]; y = U[:, -1].astype(int)
    tree = cKDTree(Z); P = tree.query_pairs(2 * eps, p=np.inf, output_type='ndarray')
    P = P[y[P[:, 0]] != y[P[:, 1]]]
    n = len(y); w = cnt / (4 * nc[y])
    m = len(P); print(label, eps, 'edges', m, flush=True)
    res = {}
    if m == 0: return dict(macro_lp=1.0, worst_lp=1.0, edges=0)
    A = sparse.csr_matrix((np.ones(2 * m), (np.repeat(np.arange(m), 2), P.ravel())), shape=(m, n))
    r1 = linprog(-w, A_ub=A, b_ub=np.ones(m), bounds=(0, 1), method='highs')
    res['macro_lp'] = -r1.fun
    # worst-class bound: max t s.t. mean_c(a) >= t
    C = sparse.csr_matrix((-cnt / nc[y], (y, np.arange(n))), shape=(4, n))
    A2 = sparse.vstack([sparse.hstack([A, sparse.csr_matrix((m, 1))]), sparse.hstack([C, sparse.csr_matrix(np.ones((4, 1)))])]).tocsr()
    c2 = np.r_[np.zeros(n), -1.0]
    r2 = linprog(c2, A_ub=A2, b_ub=np.r_[np.ones(m), np.zeros(4)], bounds=[(0, 1)] * (n + 1), method='highs')
    res['worst_lp'] = -r2.fun; res['edges'] = m
    return res

def lp():
    X, y, rid, t, itr, iv = load_all()
    Xh, yh, rh, th, _ = load_csv_dataset(D + 'raw_dataset/tep_holdout_test_data.csv')
    sd = X[itr].std(0); rows = []
    for setname, Z, yy in (('validation', X[iv], y[iv]), ('heldout', Xh, yh)):
        for coords, scl, eps in (('raw', 1.0, 0.02), ('raw', 1.0, 0.05), ('sd', sd, 0.1), ('sd', sd, 0.25), ('raw', 1.0, 0.15)):
            r = lp_bounds(Z / scl, yy, eps, setname); r.update(set=setname, coords=coords, eps=eps)
            rows.append(r); pd.DataFrame(rows).to_csv('results_r5_lp.csv', index=False); print(r, flush=True)

if __name__ == '__main__':
    {'calib': calib, 'lp': lp}[sys.argv[1]]()
