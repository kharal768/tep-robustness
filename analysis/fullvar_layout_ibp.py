"""Second variable layout and IBP certified training."""
import os
import sys, glob; mode = sys.argv[1]; sys.argv = [sys.argv[0]]
from detection_eval import *
from model import FaultClassifier
import torch.nn.functional as Fn
def layout():
    def load(prefix):
        out = []
        for f in sorted(glob.glob(f'full/{prefix}_label*_seed*.dat')):
            c = int(f.split('label')[1][0]); d = np.loadtxt(f, skiprows=1)[:, 1:53]; t = np.arange(len(d)); y = np.where((c > 0) & (t >= 40), c, 0)
            out.append((f, d, y, t))
        return out
    TR = load('tr'); HO = load('ho'); Xt = np.vstack([d for _, d, _, _ in TR]); mu, sd = Xt.mean(0), Xt.std(0)
    A = np.vstack([(d[y == 0] - mu) / sd for f, d, y, _ in TR if 'label0' in f]); st = PCAStat(A); fn = lambda z: torch.log(st.index(z))
    strip = np.zeros(52); strip[[17, 18, 49]] = 1; rows = []
    for sub, msk in (('none', None), ('all', np.ones(52)), ('stripper3', strip), ('other49', 1 - strip)):
        for f, d, y, t in HO:
            Z = (d - mu) / sd
            if msk is None: Za = torch.tensor(Z, dtype=torch.float32)
            else:
                ev = 0.3 * msk
                if (y > 0).any():
                    post = (t >= ONSET).astype(float); Za = attack_seq_vec(fn, Z, -1, ev, mask=post)
                    Za = torch.where(torch.tensor(post > 0)[:, None], Za, torch.tensor(Z, dtype=torch.float32))
                else: Za = attack_seq_vec(fn, Z, +1, ev)
            with torch.no_grad(): a = (fn(Za) > 0).numpy()
            rows.append(dict(attack=sub, run=f, cls=int(y.max()), **alarm_metrics(a, y, t)))
        R = pd.DataFrame(rows); R.to_csv('results_r19_layout.csv', index=False)
        g = R[R.attack == sub]; print(sub, 'FA', round(g[g.cls == 0].fa.mean(), 3), 'miss', g[g.cls > 0].groupby('cls').miss.mean().round(3).tolist(), flush=True)
def ibp_bounds(m, x, e):
    lo, hi = x - e, x + e
    for L in m.net:
        if isinstance(L, torch.nn.Linear):
            c, r = (hi + lo) / 2, (hi - lo) / 2; c = L(c); r = r @ L.weight.abs().T; lo, hi = c - r, c + r
        else: lo, hi = torch.relu(lo), torch.relu(hi)
    return lo, hi
def worst_logits(m, x, y, e):
    lo, hi = ibp_bounds(m, x, e); w = hi.clone(); w[torch.arange(len(y)), y] = lo[torch.arange(len(y)), y]; return w
def ibp():
    (Xtr, ytr, _), _, (Xh, yh, _) = load_sets(); F = front(); rows = []; eps = 0.02
    for s in (5000, 5001, 5002):
        hp = hp_of(F.iloc[1]); torch.manual_seed(s); m = FaultClassifier(16, 4, hidden_dim=hp.hidden_dim, depth=2)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.as_tensor(Xtr, dtype=torch.float32), torch.as_tensor(ytr)), batch_size=64, shuffle=True)
        for ep in range(50):
            f = min(1.0, ep / 20); e = eps * f; k = 1 - 0.5 * f
            for zb, yb in dl:
                opt.zero_grad(); loss = k * Fn.cross_entropy(m(zb), yb) + (1 - k) * Fn.cross_entropy(worst_logits(m, zb, yb, e), yb)
                loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval(); z = torch.as_tensor(Xh, dtype=torch.float32); yt = torch.as_tensor(yh)
        with torch.no_grad(): cert = (worst_logits(m, z, yt, eps).argmax(1) == yt).numpy()
        cp = np.array([cert[yh == c].mean() for c in range(4)]); ok = strong_ps(m, Xh, yh, eps)
        r = dict(model='ibp', seed=s, cert_macro=cp.mean(), cert_pc=json.dumps(np.round(cp, 4).tolist()), **summ(m, Xh, yh, eps, ok)); rows.append(r)
        pd.DataFrame(rows).to_csv('results_r19_ibp.csv', index=False); print(s, round(r['clean_macro'], 3), 'cert', cp.round(3), 'worst', round(r['worst_macro'], 3), r['worst_pc'], flush=True)
if __name__ == '__main__': {'layout': layout, 'ibp': ibp}[mode]()
