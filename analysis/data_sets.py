"""Builds the analysis data sets (composition blocks in ILR coordinates) from the simulated TEP runs."""
import os
import sys, os, json
from common import *
from run_pipeline import load_csv_dataset, split
torch.use_deterministic_algorithms(True)
import torchattacks as ta
HOLD = D + 'raw_dataset/tep_holdout_test_data.csv'
ONSET = 40          # sample index of fault onset (2 h at 3-min sampling)
EXCL = 20           # 60 min transition window

def filt_mask(X, y, rid, t):
    """keep rows that differ from the previous row of the same run, and drop fault samples in the transition window"""
    o = np.lexsort((t, rid)); keep = np.ones(len(y), bool)
    for a, b in zip(o[:-1], o[1:]):
        if rid[a] == rid[b] and np.array_equal(X[a], X[b]): keep[b] = False
    keep &= ~((y > 0) & (t >= ONSET) & (t < ONSET + EXCL))
    return keep

def load_sets():
    X, y, rid, t, _ = load_csv_dataset(D + 'raw_dataset/tep_multirun_composition_data.csv')
    idx = np.arange(len(y), dtype=float)[:, None]
    itr, _, iv, _ = split(idx, y, run_id=rid, t=t, val_frac=0.25, seed=0)
    itr = itr[:, 0].astype(int); iv = iv[:, 0].astype(int)
    k = filt_mask(X, y, rid, t)
    itr = itr[k[itr]]; iv = iv[k[iv]]
    Xh, yh, rh, th, _ = load_csv_dataset(HOLD); kh = filt_mask(Xh, yh, rh, th)
    return (X[itr], y[itr], rid[itr]), (X[iv], y[iv], rid[iv]), (Xh[kh], yh[kh], rh[kh])

def strong_ps(m, A, yy, eps):
    """per-sample worst case over fine PGD (100 steps, eps/20, 3 restarts), APGD-CE, APGD-DLR; full set"""
    m.eval(); z = torch.as_tensor(A, dtype=torch.float32); yt = torch.as_tensor(yy)
    ok = pred(m, z) == yy
    for rs in range(3): ok &= pred(m, pgd_attack(m, z, yt, eps, eps / 20, 100, 'linf')) == yy
    lo = float(A.min()) - 1.0; hi = float(A.max()) + 1.0; sc = hi - lo
    sm = Scaled(m, lo, sc).eval(); x = ((z - lo) / sc).reshape(-1, 16, 1, 1); e = eps / sc
    for loss in ('ce', 'dlr'):
        xa = ta.APGD(sm, norm='Linf', eps=e, steps=100, loss=loss, n_restarts=1, seed=0)(x, yt)
        ok &= pred(sm, xa) == yy
    return ok

def summ(m, A, yy, eps, ok):
    z = torch.as_tensor(A, dtype=torch.float32); yt = torch.as_tensor(yy)
    pc = rec(pred(m, z), yy); pa = rec(pred(m, pgd_attack(m, z, yt, eps, eps / 5, 10, 'linf')), yy)
    wp = np.array([ok[yy == c].mean() for c in range(4)])
    return dict(clean_macro=pc.mean(), clean_min=pc.min(), pgd10_macro=pa.mean(), pgd10_min=pa.min(),
                worst_macro=wp.mean(), worst_min=wp.min(), worst_pc=json.dumps(np.round(wp, 4).tolist()),
                clean_pc=json.dumps(np.round(pc, 4).tolist()))

if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'P':
        seeds = [int(s) for s in sys.argv[2].split(',')]
        (Xtr, ytr_, rtr), (Xv, yv_, rv), (Xh, yh, rh) = load_sets()
        print('sizes', len(ytr_), len(yv_), len(yh), flush=True)
        F = front(); sd = Xtr.std(0)
        out = f'results_r5_P_{seeds[0]}.csv'; rows = []
        os.makedirs('models', exist_ok=True); os.makedirs('persample', exist_ok=True)
        tuned = Hyperparams(lr=8.596254586061327e-4, weight_decay=1.2889511341131778e-06, hidden_dim=113, depth=2, lambda_adv=1.1224722951166286, pgd_steps_train=8)
        for s in seeds:
            for key, (coords, scl, eps) in {'raw002': ('raw', 1.0, 0.02), 'sd01': ('sd', sd, 0.1), 'sd025': ('sd', sd, 0.25)}.items():
                A_tr, A_v, A_h = Xtr / scl, Xv / scl, Xh / scl
                specs = [('standard', hp_of(F.iloc[1]), eps / 5), ('adv_pgd4', hp_of(F.iloc[3]), eps / 5)]
                h10 = hp_of(F.iloc[3]); h10.pgd_steps_train = 10; specs.append(('adv_pgd10', h10, eps / 4))
                if key != 'sd01': specs.append(('tuned', tuned, eps / 5))
                for tag, hp, alpha in specs:
                    c = EvalConfig(eps=eps, pgd_alpha=alpha, pgd_steps_eval=10)
                    m, _ = train_model(hp, A_tr, ytr_, A_v, yv_, 4, c, epochs=50, seed=s); m.eval()
                    torch.save(m.state_dict(), f'models/{key}_{tag}_{s}.pt')
                    for split_name, A, yy, rr in (('val', A_v, yv_, rv), ('heldout', A_h, yh, rh)):
                        ok = strong_ps(m, A, yy, eps)
                        if split_name == 'heldout': np.savez(f'persample/{key}_{tag}_{s}.npz', ok=ok, y=yy, rid=rr)
                        r = dict(set=split_name, budget=key, coords=coords, eps=eps, model=tag, seed=s, **summ(m, A, yy, eps, ok))
                        rows.append(r); pd.DataFrame(rows).to_csv(out, index=False)
                        print(split_name, key, tag, s, round(r['clean_macro'], 3), round(r['worst_macro'], 3), round(r['worst_min'], 3), flush=True)
