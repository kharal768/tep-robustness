"""Matched-FAR detection with PCA T2/SPE, EWMA and LDA monitors under attack (P4)."""
import os
import sys, json
from data_sets import *
from detection_cnn import runs, ewma_t, PCAStat, alarm_metrics, CNN, pgd_w
import torch.nn.functional as Fn
from model import FaultClassifier

def full_attacks(m, A, yy, eps):
    z = torch.as_tensor(A, dtype=torch.float32); yt = torch.as_tensor(yy); out = {}
    pc = pred(m, z); out['clean'] = pc == yy
    out['pgd10'] = pred(m, pgd_attack(m, z, yt, eps, eps / 5, 10, 'linf')) == yy
    ok = out['clean'].copy()
    for r in range(3): ok &= pred(m, pgd_attack(m, z, yt, eps, eps / 20, 100, 'linf')) == yy
    out['pgd_fine'] = ok.copy()
    lo = float(A.min()) - 1.0; hi = float(A.max()) + 1.0; sc = hi - lo
    sm = Scaled(m, lo, sc).eval(); x = ((z - lo) / sc).reshape(-1, 16, 1, 1); e = eps / sc
    padv = None
    for nm, atk in (('apgd_ce', ta.APGD(sm, norm='Linf', eps=e, steps=50, loss='ce', n_restarts=1, seed=0)),
                    ('apgd_dlr', ta.APGD(sm, norm='Linf', eps=e, steps=50, loss='dlr', n_restarts=1, seed=0)),
                    ('square', ta.Square(sm, norm='Linf', eps=e, n_queries=1000, n_restarts=1, seed=0))):
        p = pred(sm, atk(x, yt)); out[nm] = p == yy
        if nm == 'apgd_ce': padv = p
    out['worst'] = out['clean'] & out['pgd_fine'] & out['apgd_ce'] & out['apgd_dlr'] & out['square']
    return out, padv

def stage_E(seeds):
    (Xtr, ytr_, rtr), (Xv, yv_, rv), (Xh, yh, rh) = load_sets(); F = front(); rows = []
    cfg = EvalConfig(eps=0.15, pgd_alpha=0.03, pgd_steps_eval=10); out = f'results_r6_E_{seeds[0]}.csv'
    for s in seeds:
        for i in range(8):
            r = F.iloc[i]; m, _ = train_model(hp_of(r), Xtr, ytr_, Xv, yv_, 4, cfg, epochs=50, seed=s); m.eval()
            for setn, A, yy in (('val', Xv, yv_), ('heldout', Xh, yh)):
                o, padv = full_attacks(m, A, yy, 0.15)
                row = dict(set=setn, config=i, lambda_adv=r.lambda_adv, seed=s)
                for k, v in o.items():
                    pcs = [v[yy == c].mean() for c in range(4)]; row[k + '_macro'] = float(np.mean(pcs)); row[k + '_min'] = float(np.min(pcs))
                row['worst_pc'] = json.dumps(np.round([o['worst'][yy == c].mean() for c in range(4)], 4).tolist())
                row['conf_apgd'] = json.dumps([[float((padv[yy == c] == k).mean()) for k in range(4)] for c in range(4)])
                rows.append(row); pd.DataFrame(rows).to_csv(out, index=False)
                print(setn, i, s, round(row['clean_macro'], 3), round(row['pgd10_macro'], 3), round(row['worst_macro'], 3), flush=True)

class LDA:
    def __init__(s, A, y):
        mu = np.array([A[y == c].mean(0) for c in range(4)]); Sw = sum(np.cov((A[y == c] - mu[c]).T) * (np.sum(y == c) - 1) for c in range(4)) / (len(y) - 4)
        Si = np.linalg.inv(Sw + 1e-9 * np.eye(A.shape[1])); pri = np.log(np.bincount(y, minlength=4) / len(y))
        s.W = torch.tensor(Si @ mu.T, dtype=torch.float32); s.b = torch.tensor(-0.5 * np.einsum('ij,jk,ik->i', mu, Si, mu) + pri, dtype=torch.float32)
    def __call__(s, z): return z @ s.W + s.b

def attack_seq_vec(score_fn, Z, sign, epsv, steps=100, restarts=3, mask=None):
    """PGD on a whole run with per-coordinate budget epsv; maximises sign * sum_t mask_t * score_t"""
    z0 = torch.as_tensor(Z, dtype=torch.float32); ev = torch.as_tensor(epsv, dtype=torch.float32); best = None; bestv = None
    mk = torch.ones(len(Z)) if mask is None else torch.as_tensor(mask, dtype=torch.float32)
    for r in range(restarts):
        g = torch.Generator().manual_seed(r); d = (torch.rand(z0.shape, generator=g) * 2 - 1) * ev
        for _ in range(steps):
            d.requires_grad_(True); v = sign * (score_fn(z0 + d) * mk).sum(); gr, = torch.autograd.grad(v, d)
            d = torch.max(torch.min(d.detach() + ev / 20 * gr.sign(), ev), -ev)
        with torch.no_grad(): v = sign * (score_fn(z0 + d) * mk).sum().item()
        if bestv is None or v > bestv: bestv, best = v, d.detach()
    return z0 + best

def stage_Dm():
    X, y, rid, t, _ = load_csv_dataset(D + 'raw_dataset/tep_multirun_composition_data.csv')
    idx = np.arange(len(y), dtype=float)[:, None]; itr, _, iv, _ = split(idx, y, run_id=rid, t=t, val_frac=0.25, seed=0)
    itr = itr[:, 0].astype(int); iv = iv[:, 0].astype(int)
    (Xtr_a, ytr_a, _), _, _ = load_sets(); sd = Xtr_a.std(0)
    Atr = X[itr][y[itr] == 0]
    Asm = np.vstack([ewma_t(torch.tensor(Z, dtype=torch.float32)).numpy()[yy == 0] for _, Z, yy, _ in runs(X[itr], y[itr], rid[itr], t[itr])])
    st = PCAStat(Atr); se = PCAStat(Asm); lda = LDA(X[itr], y[itr])
    V = runs(X[iv], y[iv], rid[iv], t[iv]); H = runs(*load_csv_dataset(HOLD)[:4])
    F = front(); tuned = Hyperparams(lr=8.596254586061327e-4, weight_decay=1.2889511341131778e-06, hidden_dim=113, depth=2, lambda_adv=1.1224722951166286, pgd_steps_train=8)
    rows = []
    for budget, scl, eps in (('raw002', 1.0, 0.02), ('sd025', sd, 0.25)):
        epsv = eps * (np.ones(16) if budget == 'raw002' else sd)
        dets = {'PCA T2/SPE': lambda z: torch.log(st.index(z)), 'EWMA PCA T2/SPE': lambda z: torch.log(se.index(ewma_t(z))),
                'LDA': lambda z: torch.logsumexp(lda(z)[:, 1:], 1) - lda(z)[:, 0]}
        for tag, hp in (('standard', hp_of(F.iloc[1])), ('adv_pgd10', hp_of(F.iloc[3])), ('tuned', tuned)):
            for s in range(5000, 5005):
                m = FaultClassifier(16, 4, hidden_dim=hp.hidden_dim, depth=2); m.load_state_dict(torch.load(f'models/{budget}_{tag}_{s}.pt')); m.eval()
                sc_ = torch.as_tensor(scl, dtype=torch.float32)
                dets[f'net:{tag}:{s}'] = (lambda m, sc_: (lambda z: torch.logsumexp(m(z / sc_)[:, 1:], 1) - m(z / sc_)[:, 0]))(m, sc_)
        for name, fn in dets.items():
            # threshold: 99th percentile of the clean score on validation normal-labelled samples (1% false-alarm rate)
            with torch.no_grad():
                sv = np.concatenate([fn(torch.tensor(Z, dtype=torch.float32)).numpy()[yy == 0] for _, Z, yy, _ in V])
            thr = float(np.quantile(sv, 0.99))
            for r, Z, yy, tt in H:
                with torch.no_grad(): a0 = (fn(torch.tensor(Z, dtype=torch.float32)) > thr).numpy()
                if (yy > 0).any():
                    post = (tt >= ONSET).astype(float); Za = attack_seq_vec(fn, Z, -1, epsv, mask=post)
                    Za = torch.where(torch.tensor(post > 0)[:, None], Za, torch.tensor(Z, dtype=torch.float32))
                else:
                    Za = attack_seq_vec(fn, Z, +1, epsv)
                with torch.no_grad(): a1 = (fn(Za) > thr).numpy()
                for cond, a in (('clean', a0), ('attacked', a1)):
                    rows.append(dict(budget=budget, detector=name, run=r, cls=int(yy.max()), cond=cond, **alarm_metrics(a, yy, tt)))
            pd.DataFrame(rows).to_csv('results_r6_Dm.csv', index=False); print(budget, name, flush=True)

def windows_f(X, y, rid, t, w=10):
    Wl, Yl, Rl = [], [], []
    for r in np.unique(rid):
        k = np.where(rid == r)[0]; k = k[np.argsort(t[k])]
        for j in range(w - 1, len(k)):
            lab = y[k[j]]; tt = t[k[j]]
            if np.array_equal(X[k[j]], X[k[j - 1]]): continue          # last sample is a held duplicate
            if lab > 0 and ONSET <= tt < ONSET + EXCL: continue
            if lab > 0 and t[k[j - w + 1]] < ONSET: continue
            Wl.append(X[k[j - w + 1:j + 1]].T); Yl.append(lab); Rl.append(r)
    return np.array(Wl, dtype=np.float32), np.array(Yl), np.array(Rl)

def stage_W5():
    X, y, rid, t, _ = load_csv_dataset(D + 'raw_dataset/tep_multirun_composition_data.csv')
    idx = np.arange(len(y), dtype=float)[:, None]; itr, _, iv, _ = split(idx, y, run_id=rid, t=t, val_frac=0.25, seed=0)
    tr_runs = set(rid[itr[:, 0].astype(int)]); v_runs = set(rid[iv[:, 0].astype(int)]) - tr_runs
    mtr = np.isin(rid, list(tr_runs)); mv = np.isin(rid, list(v_runs))
    Wtr, Ytr, _ = windows_f(X[mtr], y[mtr], rid[mtr], t[mtr]); Wv, Yv, _ = windows_f(X[mv], y[mv], rid[mv], t[mv])
    Xh, yh, rh, th, _ = load_csv_dataset(HOLD); Wh, Yh, Rh = windows_f(Xh, yh, rh, th)
    print('windows', len(Ytr), len(Yv), len(Yh), flush=True); eps = 0.02; rows = []
    for s in range(5000, 5005):
        for tag, lam in (('standard', 0.0), ('adversarial', 0.55)):
            torch.manual_seed(s); m = CNN(); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(Wtr), torch.tensor(Ytr)), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(s))
            for ep in range(50):
                m.train()
                for xb, yb in dl:
                    loss = Fn.cross_entropy(m(xb), yb)
                    if lam > 0: loss = loss + lam * Fn.cross_entropy(m(pgd_w(m, xb, yb, eps, eps / 5, 4)), yb)
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            m.eval()
            for setn, W, Y, R in (('val', Wv, Yv, None), ('heldout', Wh, Yh, Rh)):
                x = torch.tensor(W); yt = torch.tensor(Y); ok = pred(m, x) == Y; pc = rec(pred(m, x), Y)
                for r_ in range(3): ok &= pred(m, pgd_w(m, x, yt, eps, eps / 20, 100)) == Y
                lo = float(W.min()) - 1.0; hi = float(W.max()) + 1.0; sc = hi - lo; sm = Scaled(m, lo, sc).eval(); xs = ((x - lo) / sc).reshape(-1, 16, 10, 1)
                for loss in ('ce', 'dlr'): ok &= pred(sm, ta.APGD(sm, norm='Linf', eps=eps / sc, steps=100, loss=loss, n_restarts=1, seed=0)(xs, yt)) == Y
                if R is not None: np.savez(f'persample/cnn_{tag}_{s}.npz', ok=ok, y=Y, rid=R)
                wp = np.array([ok[Y == c].mean() for c in range(4)])
                rows.append(dict(set=setn, model=tag, seed=s, clean_macro=pc.mean(), clean_min=pc.min(), worst_macro=wp.mean(), worst_min=wp.min(),
                                 worst_pc=json.dumps(np.round(wp, 4).tolist()), clean_pc=json.dumps(np.round(pc, 4).tolist())))
                pd.DataFrame(rows).to_csv('results_r6_W5.csv', index=False); print(setn, tag, s, round(wp.mean(), 3), np.round(wp, 3), flush=True)

if __name__ == '__main__':
    st_ = sys.argv[1]
    if st_ == 'E': stage_E([int(x) for x in sys.argv[2].split(',')])
    elif st_ == 'Dm': stage_Dm()
    elif st_ == 'W5': stage_W5()
