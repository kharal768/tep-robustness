"""Trains the CNN fault classifier used for the matched-FAR detection comparison."""
import os
import sys, json
from common import *
from run_pipeline import load_csv_dataset, split
import torchattacks as ta
import torch.nn as nn, torch.nn.functional as Fn
torch.use_deterministic_algorithms(True)
HOLD = D + 'raw_dataset/tep_holdout_test_data.csv'
ONSET, EXCL = 40, 20

def runs(X, y, rid, t):
    out = []
    for r in np.unique(rid):
        k = np.where(rid == r)[0]; k = k[np.argsort(t[k])]
        out.append((r, X[k], y[k], t[k]))
    return out

def ewma_t(Z, lam=0.2):
    out = [Z[0]]
    for i in range(1, Z.shape[0]): out.append(lam * Z[i] + (1 - lam) * out[-1])
    return torch.stack(out)

class PCAStat:
    def __init__(s, A):
        mu = A.mean(0); U, S, Vt = np.linalg.svd(A - mu, full_matrices=False); ev = S**2 / (len(A) - 1)
        k = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.95) + 1)
        s.mu = torch.tensor(mu, dtype=torch.float32); s.P = torch.tensor(Vt[:k].T, dtype=torch.float32); s.lam = torch.tensor(ev[:k] + 1e-12, dtype=torch.float32)
        T2, SPE = s.raw(torch.tensor(A, dtype=torch.float32))
        s.l1 = float(np.quantile(T2.numpy(), .99)); s.l2 = float(np.quantile(SPE.numpy(), .99)); s.k = k
    def raw(s, z):
        d = z - s.mu; sc = d @ s.P; return (sc**2 / s.lam).sum(-1), ((d - sc @ s.P.T)**2).sum(-1)
    def index(s, z):
        T2, SPE = s.raw(z); return torch.maximum(T2 / s.l1, SPE / s.l2)

def alarm_metrics(alarm, y, t):
    """alarm: bool per sample of one run (time-ordered)"""
    out = {}
    nm = (y == 0); out['fa'] = alarm[nm].mean() if nm.any() else np.nan
    if (y > 0).any():
        post = (t >= ONSET + EXCL); out['miss'] = 1 - alarm[(y > 0) & post].mean()
        # delay: first time after onset with 3 consecutive alarms (minutes)
        idx = np.where(t >= ONSET)[0]; a = alarm[idx]; d = np.nan
        for j in range(len(a) - 2):
            if a[j] and a[j + 1] and a[j + 2]: d = (t[idx[j]] - ONSET) * 3.0; break
        out['delay'] = d
    return out

def attack_seq(score_fn, Z, sign, eps, steps=100, restarts=3):
    """PGD on a whole run: sign=-1 lowers sum(score) (evasion), +1 raises it (false alarms). score_fn maps (T,16)->(T,)"""
    best = None; bestv = None; z0 = torch.as_tensor(Z, dtype=torch.float32)
    for r in range(restarts):
        g = torch.Generator().manual_seed(r)
        d = (torch.rand(z0.shape, generator=g) * 2 - 1) * eps
        for _ in range(steps):
            d.requires_grad_(True); v = sign * score_fn(z0 + d).sum(); gr, = torch.autograd.grad(v, d)
            d = (d.detach() + eps / 20 * gr.sign()).clamp(-eps, eps)
        with torch.no_grad(): v = sign * score_fn(z0 + d).sum().item()
        if bestv is None or v > bestv: bestv, best = v, d.detach()
    return z0 + best

def stage_D():
    X, y, rid, t, _ = load_csv_dataset(D + 'raw_dataset/tep_multirun_composition_data.csv')
    idx = np.arange(len(y), dtype=float)[:, None]; itr, _, iv, _ = split(idx, y, run_id=rid, t=t, val_frac=0.25, seed=0); itr = itr[:, 0].astype(int)
    Atr = X[itr][y[itr] == 0]
    # EWMA-smoothed normal training data
    tr_runs = runs(X[itr], y[itr], rid[itr], t[itr])
    Asm = np.vstack([ewma_t(torch.tensor(Z, dtype=torch.float32)).numpy()[yy == 0] for _, Z, yy, _ in tr_runs])
    st = PCAStat(Atr); se = PCAStat(Asm)
    Xh, yh, rh, th, _ = load_csv_dataset(HOLD); H = runs(Xh, yh, rh, th)
    from model import FaultClassifier
    F = front(); rows = []; eps = 0.02
    nets = {}
    for tag, hp in (('standard', hp_of(F.iloc[1])), ('adv_pgd10', hp_of(F.iloc[3])),
                    ('tuned', Hyperparams(lr=8.596e-4, weight_decay=1.289e-6, hidden_dim=113, depth=2, lambda_adv=1.1225, pgd_steps_train=8))):
        for s in (5000, 5001, 5002):
            m = FaultClassifier(16, 4, hidden_dim=hp.hidden_dim, depth=2); m.load_state_dict(torch.load(f'models/raw002_{tag}_{s}.pt')); m.eval()
            nets[(tag, s)] = m
    dets = {'PCA T2/SPE': lambda z: st.index(z), 'EWMA PCA T2/SPE': lambda z: se.index(ewma_t(z))}
    for (tag, s), m in nets.items():
        dets[f'net:{tag}:{s}'] = (lambda m: (lambda z: torch.logsumexp(m(z)[:, 1:], 1) - m(z)[:, 0]))(m)   # >0 means alarm (fault more likely than normal)
    for name, fn in dets.items():
        thr = 1.0 if 'PCA' in name else 0.0
        for r, Z, yy, tt in H:
            with torch.no_grad(): a0 = (fn(torch.tensor(Z, dtype=torch.float32)) > thr).numpy()
            fault_run = (yy > 0).any()
            if fault_run:
                # attacker hides the fault after onset; pre-onset samples untouched
                post = torch.tensor(tt >= ONSET)
                fn2 = (lambda fn, post: (lambda z: fn(z) * post))(fn, post)
                Za = attack_seq(fn2, Z, -1, eps) if True else None
                Za = torch.where(post[:, None], Za, torch.tensor(Z, dtype=torch.float32))
            else:
                Za = attack_seq(fn, Z, +1, eps)
            with torch.no_grad(): a1 = (fn(Za) > thr).numpy()
            cls = int(yy.max())
            for cond, a in (('clean', a0), ('attacked', a1)):
                rows.append(dict(detector=name, run=r, cls=cls, cond=cond, **alarm_metrics(a, yy, tt)))
        pd.DataFrame(rows).to_csv('results_r5_D.csv', index=False); print(name, flush=True)

def stage_C():
    Ztr, ytr, Zv, yv = load(); F = front(); cfg = EvalConfig(eps=0.15, pgd_alpha=0.03, pgd_steps_eval=10); rows = []
    for i in range(5):
        r = F.iloc[i]; m, _ = train_model(hp_of(r), Ztr, ytr, Zv, yv, 4, cfg, epochs=50, seed=5000); m.eval()
        z = torch.as_tensor(Zv, dtype=torch.float32); yt = torch.as_tensor(yv)
        lo = float(Zv.min()) - 1.0; hi = float(Zv.max()) + 1.0; sc = hi - lo; sm = Scaled(m, lo, sc).eval(); x = ((z - lo) / sc).reshape(-1, 16, 1, 1)
        for nm, atk in (('pgd10', None), ('apgd_ce', ta.APGD(sm, norm='Linf', eps=0.15 / sc, steps=50, loss='ce', n_restarts=1, seed=0)),
                        ('square', ta.Square(sm, norm='Linf', eps=0.15 / sc, n_queries=1000, n_restarts=1, seed=0))):
            p = pred(m, pgd_attack(m, z, yt, 0.15, 0.03, 10, 'linf')) if atk is None else pred(sm, atk(x, yt))
            for c in range(4):
                rows.append(dict(config=i, lambda_adv=r.lambda_adv, attack=nm, true=c, **{f'pred{k}': float((p[yv == c] == k).mean()) for k in range(4)}))
        pc = pred(m, z)
        for c in range(4): rows.append(dict(config=i, lambda_adv=r.lambda_adv, attack='clean', true=c, **{f'pred{k}': float((pc[yv == c] == k).mean()) for k in range(4)}))
        pd.DataFrame(rows).to_csv('results_r5_C.csv', index=False); print(i, flush=True)

class CNN(nn.Module):
    def __init__(s, w=10):
        super().__init__(); s.w = w
        s.c1 = nn.Conv1d(16, 64, 3, padding=1); s.c2 = nn.Conv1d(64, 64, 3, padding=1); s.fc = nn.Linear(64, 4)
    def forward(s, x):
        x = x.reshape(x.shape[0], 16, s.w); h = Fn.relu(s.c2(Fn.relu(s.c1(x)))); return s.fc(h.mean(-1))

def windows(X, y, rid, t, w=10):
    W, Y, R = [], [], []
    for r in np.unique(rid):
        k = np.where(rid == r)[0]; k = k[np.argsort(t[k])]
        for j in range(w - 1, len(k)):
            lab = y[k[j]]; tt = t[k[j]]
            if lab > 0 and ONSET <= tt < ONSET + EXCL: continue
            if lab > 0 and t[k[j - w + 1]] < ONSET: continue   # window must lie after onset for fault labels
            W.append(X[k[j - w + 1:j + 1]].T); Y.append(lab); R.append(r)
    return np.array(W, dtype=np.float32), np.array(Y), np.array(R)

def pgd_w(m, x, y, eps, alpha, steps, rand=True):
    d = (torch.rand_like(x) * 2 - 1) * eps if rand else torch.zeros_like(x)
    for _ in range(steps):
        d.requires_grad_(True); l = Fn.cross_entropy(m(x + d), y); g, = torch.autograd.grad(l, d)
        d = (d.detach() + alpha * g.sign()).clamp(-eps, eps)
    return (x + d).detach()

def stage_W():
    X, y, rid, t, _ = load_csv_dataset(D + 'raw_dataset/tep_multirun_composition_data.csv')
    idx = np.arange(len(y), dtype=float)[:, None]; itr, _, iv, _ = split(idx, y, run_id=rid, t=t, val_frac=0.25, seed=0)
    itr = itr[:, 0].astype(int); iv = iv[:, 0].astype(int)
    # a fault run contributes its pre-onset part to class 0; keep whole runs on one side (runs are assigned by the split)
    tr_runs = set(rid[itr]); v_runs = set(rid[iv]) - tr_runs
    mtr = np.isin(rid, list(tr_runs)); mv = np.isin(rid, list(v_runs))
    Wtr, Ytr, _ = windows(X[mtr], y[mtr], rid[mtr], t[mtr]); Wv, Yv, _ = windows(X[mv], y[mv], rid[mv], t[mv])
    Xh, yh, rh, th, _ = load_csv_dataset(HOLD); Wh, Yh, Rh = windows(Xh, yh, rh, th)
    print('windows', len(Ytr), len(Yv), len(Yh), flush=True)
    eps = 0.02; rows = []
    for s in (5000, 5001, 5002):
        for tag, lam in (('standard', 0.0), ('adversarial', 0.55)):
            torch.manual_seed(s); m = CNN(); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(Wtr), torch.tensor(Ytr)), batch_size=64, shuffle=True,
                                             generator=torch.Generator().manual_seed(s))
            for ep in range(30):
                m.train()
                for xb, yb in dl:
                    loss = Fn.cross_entropy(m(xb), yb)
                    if lam > 0: loss = loss + lam * Fn.cross_entropy(m(pgd_w(m, xb, yb, eps, eps / 5, 4)), yb)
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            m.eval()
            for setn, W, Y in (('val', Wv, Yv), ('heldout', Wh, Yh)):
                x = torch.tensor(W); yt = torch.tensor(Y)
                ok = pred(m, x) == Y; pc = rec(pred(m, x), Y)
                for r_ in range(3): ok &= pred(m, pgd_w(m, x, yt, eps, eps / 20, 100)) == Y
                lo = float(W.min()) - 1.0; hi = float(W.max()) + 1.0; sc = hi - lo
                sm = Scaled(m, lo, sc).eval(); xs = ((x - lo) / sc).reshape(-1, 16, 10, 1)
                for loss in ('ce', 'dlr'):
                    ok &= pred(sm, ta.APGD(sm, norm='Linf', eps=eps / sc, steps=100, loss=loss, n_restarts=1, seed=0)(xs, yt)) == Y
                wp = np.array([ok[Y == c].mean() for c in range(4)])
                rows.append(dict(set=setn, model=tag, seed=s, clean_macro=pc.mean(), clean_min=pc.min(), worst_macro=wp.mean(), worst_min=wp.min(),
                                 worst_pc=json.dumps(np.round(wp, 4).tolist()), clean_pc=json.dumps(np.round(pc, 4).tolist())))
                pd.DataFrame(rows).to_csv('results_r5_W.csv', index=False); print(rows[-1], flush=True)

if __name__ == '__main__':
    {'D': stage_D, 'C': stage_C, 'W': stage_W}[sys.argv[1]]()
