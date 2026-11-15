"""SD-unit calibration, run-density curve and rotated ILR bases."""
import os
import sys, time
from data_sets import *
from calibration_bounds import inv, BL, load_all
from bounds_centre_lp import solve
from model import FaultClassifier
def sdcal():
    (Xtr, _, _), _, _ = load_sets(); sd = Xtr.std(0)
    X, y, rid, t, itr, iv = load_all(); nom = np.array([str(r).startswith('label0') for r in rid]) & np.isin(np.arange(len(y)), itr)
    cal = pd.read_csv('results_r5_calib.csv'); P = cal[cal.kind == 'part'].reset_index(drop=True); zbar = X[nom].mean(0); rows = []
    for a, b, name, parts in BL:
        zb = zbar[a:b]; x0 = inv(zb)
        for i, p in enumerate(parts):
            xn = float(P[(P.block == name) & (P.part == p)].noise_share.iloc[0]); out = dict(block=name, part=p, share=x0[i], noise_share=xn)
            g = np.array([(inv(zb + 1e-6 * np.eye(len(zb))[k])[i] - x0[i]) / 1e-6 for k in range(len(zb))])
            for kk in (0.1, 0.25):
                ev = kk * sd[a:b]; best = max(abs(inv(zb + s * ev * np.sign(g))[i] - x0[i]) for s in (1, -1))
                out[f'max_change_sd{kk}'] = best; out[f'ratio_sd{kk}'] = best / xn
            rows.append(out)
    R = pd.DataFrame(rows); R.to_csv('results_r12_sdcal.csv', index=False); print(R[R.share > 0.05].round(4).to_string())
def nruns():
    _, _, (Xh, yh, rh) = load_sets(); rows = []; rng = np.random.default_rng(0)
    for k in (3, 5, 7, 10):
        for d in range(3 if k < 10 else 1):
            keep = []
            for c in range(4):
                runs_c = np.unique(rh[yh == c]) if c > 0 else np.unique(rh[[str(r).startswith('holdout_label0') for r in rh]])
                keep += list(rng.choice(runs_c, size=min(k, len(runs_c)), replace=False))
            # normal samples come from all selected runs (pre-onset samples of fault runs included)
            m = np.isin(rh, keep); r = solve(Xh[m], yh[m], 0.05, centres=True)
            rows.append(dict(runs_per_class=k, draw=d, n=int(m.sum()), macro=r['macro'], worst=r['worst'], edges=r['edges']))
            pd.DataFrame(rows).to_csv('results_r12_nruns.csv', index=False); print(rows[-1], flush=True)
def rotq(seed):
    rng = np.random.default_rng(seed); Q = np.zeros((16, 16))
    for a, b, _, _ in BL:
        q, _ = np.linalg.qr(rng.standard_normal((b - a, b - a))); Q[a:b, a:b] = q
    return Q
def rot():
    (Xtr, ytr, _), (Xv, yv, _), (Xh, yh, _) = load_sets(); F = front(); rows = []
    for rs in (1, 2):
        Q = rotq(rs); A, B, C = Xtr @ Q, Xv @ Q, Xh @ Q
        b = solve(C, yh, 0.02, centres=True); rows.append(dict(basis=rs, model='bound', seed=-1, worst_macro=b['macro'], worst_min=b['worst']))
        for s in (5000, 5001, 5002):
            h10 = hp_of(F.iloc[3]); h10.pgd_steps_train = 10
            for tag, hp, alpha in (('standard', hp_of(F.iloc[1]), 0.004), ('adv_pgd10', h10, 0.005)):
                c = EvalConfig(eps=0.02, pgd_alpha=alpha, pgd_steps_eval=10)
                m, _ = train_model(hp, A, ytr, B, yv, 4, c, epochs=50, seed=s); m.eval()
                ok = strong_ps(m, C, yh, 0.02); r = dict(basis=rs, model=tag, seed=s, **summ(m, C, yh, 0.02, ok)); rows.append(r)
                pd.DataFrame(rows).to_csv('results_r12_rot.csv', index=False); print(rs, tag, s, round(r['clean_macro'], 3), round(r['worst_macro'], 3), r['worst_pc'], flush=True)
if __name__ == '__main__': {'sdcal': sdcal, 'nruns': nruns, 'rot': rot}[sys.argv[1]]()
