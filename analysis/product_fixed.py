"""Reruns with the product block held fixed."""
import os
import sys; sys.argv0 = sys.argv[1]; sys.argv = [sys.argv[0]]
from detection_eval import *
from model import FaultClassifier
from bounds_centre_lp import solve
SIG = pd.read_csv('results_r5_calib.csv').query("kind=='coord'").sort_values('coord').noise_sd.values
EPSV = 2 * SIG.copy(); EPSV[12:] = 0.0
def load(tag, s, pre):
    F = front(); hp = hp_of(F.iloc[1]) if tag == 'standard' else hp_of(F.iloc[3])
    m = FaultClassifier(16, 4, hidden_dim=hp.hidden_dim, depth=2); m.load_state_dict(torch.load(f'models/{pre}_{tag}_{s}.pt')); m.eval(); return m
def wrap(m, pre):
    sg = torch.as_tensor(SIG, dtype=torch.float32)
    return m if pre == 'raw002' else (lambda z: m(z / sg))
def vpgd(f, z, y, ev, steps=200, restarts=3):
    ok = (f(z).argmax(1) == y).numpy()
    for r in range(restarts):
        g = torch.Generator().manual_seed(r); d = (torch.rand(z.shape, generator=g) * 2 - 1) * ev
        for _ in range(steps):
            d.requires_grad_(True); L = Fn.cross_entropy(f(z + d), y); gr, = torch.autograd.grad(L, d)
            d = torch.max(torch.min(d.detach() + ev / 20 * gr.sign(), ev), -ev)
        with torch.no_grad(): ok &= (f(z + d).argmax(1) == y).numpy()
    return ok
def bound():
    _, _, (Xh, yh, _) = load_sets(); sc = SIG.copy(); sc[12:] = 1e-6
    r = solve(Xh / sc, yh, 2.0, centres=True); r0 = solve(Xh / sc, yh, 2.0, centres=False); print(r, r0)
    pd.DataFrame([dict(pair=r0['macro'], centre=r['macro'], centre_worst=r['worst'], edges=r['edges'])]).to_csv('results_r15_bound.csv', index=False)
def cls():
    _, _, (Xh, yh, _) = load_sets(); z = torch.as_tensor(Xh, dtype=torch.float32); yt = torch.as_tensor(yh); ev = torch.as_tensor(EPSV, dtype=torch.float32); rows = []
    for pre, seeds in (('raw002', range(5000, 5005)), ('noise2', range(5000, 5003))):
        for tag in ('standard', 'adv_pgd10'):
            for s in seeds:
                f = wrap(load(tag, s, pre), pre)
                with torch.no_grad(): pc = rec(f(z).argmax(1).numpy(), yh)
                ok = vpgd(f, z, yt, ev); wp = np.array([ok[yh == c].mean() for c in range(4)])
                rows.append(dict(models=pre, model=tag, seed=s, clean_macro=pc.mean(), worst_macro=wp.mean(), worst_min=wp.min(), worst_pc=json.dumps(np.round(wp, 4).tolist())))
                pd.DataFrame(rows).to_csv('results_r15_cls.csv', index=False); print(pre, tag, s, round(pc.mean(), 3), round(wp.mean(), 3), wp.round(2), flush=True)
def gnoise():
    _, _, (Xh, yh, _) = load_sets(); rng = np.random.default_rng(0); sg = SIG.copy(); sg[12:] = 0; rows = []
    for tag in ('standard', 'adv_pgd10'):
        for s in range(5000, 5005):
            m = load(tag, s, 'raw002')
            for k in (0.0, 1.0, 2.0):
                pcs = [rec(pred(m, torch.as_tensor(Xh + k * sg * rng.standard_normal(Xh.shape), dtype=torch.float32)), yh) for _ in range(10 if k else 1)]
                pc = np.mean(pcs, 0); rows.append(dict(model=tag, seed=s, k=k, macro=pc.mean(), pc=json.dumps(np.round(pc, 4).tolist())))
    R = pd.DataFrame(rows); R.to_csv('results_r15_gnoise.csv', index=False); print(R.groupby(['model', 'k']).macro.mean().round(3))
def det():
    X, y, rid, t, _ = load_csv_dataset(D + 'raw_dataset/tep_multirun_composition_data.csv')
    idx = np.arange(len(y), dtype=float)[:, None]; itr, _, iv, _ = split(idx, y, run_id=rid, t=t, val_frac=0.25, seed=0)
    itr = itr[:, 0].astype(int); iv = iv[:, 0].astype(int)
    Atr = X[itr][y[itr] == 0]
    Asm = np.vstack([ewma_t(torch.tensor(Z, dtype=torch.float32)).numpy()[yy == 0] for _, Z, yy, _ in runs(X[itr], y[itr], rid[itr], t[itr])])
    st = PCAStat(Atr); se = PCAStat(Asm); lda = LDA(X[itr], y[itr])
    V = runs(X[iv], y[iv], rid[iv], t[iv]); H = runs(*load_csv_dataset(HOLD)[:4])
    dets = {'PCA T2/SPE': lambda z: torch.log(st.index(z)), 'EWMA PCA T2/SPE': lambda z: torch.log(se.index(ewma_t(z))),
            'LDA': lambda z: torch.logsumexp(lda(z)[:, 1:], 1) - lda(z)[:, 0]}
    for pre in ('raw002', 'noise2'):
        for tag in ('standard', 'adv_pgd10'):
            for s in (5000, 5001, 5002):
                f = wrap(load(tag, s, pre), pre)
                dets[f'net:{pre}:{tag}:{s}'] = (lambda f: (lambda z: torch.logsumexp(f(z)[:, 1:], 1) - f(z)[:, 0]))(f)
    rows = []
    for name, fn in dets.items():
        with torch.no_grad(): sv = np.concatenate([fn(torch.tensor(Z, dtype=torch.float32)).numpy()[yy == 0] for _, Z, yy, _ in V])
        thr = float(np.quantile(sv, 0.99))
        for r, Z, yy, tt in H:
            with torch.no_grad(): a0 = (fn(torch.tensor(Z, dtype=torch.float32)) > thr).numpy()
            if (yy > 0).any():
                post = (tt >= ONSET).astype(float); Za = attack_seq_vec(fn, Z, -1, EPSV, mask=post)
                Za = torch.where(torch.tensor(post > 0)[:, None], Za, torch.tensor(Z, dtype=torch.float32))
            else: Za = attack_seq_vec(fn, Z, +1, EPSV)
            with torch.no_grad(): a1 = (fn(Za) > thr).numpy()
            for cond, a in (('clean', a0), ('attacked', a1)): rows.append(dict(detector=name, run=r, cls=int(yy.max()), cond=cond, **alarm_metrics(a, yy, tt)))
        pd.DataFrame(rows).to_csv('results_r15_det.csv', index=False); print(name, flush=True)
if __name__ == '__main__': {'bound': bound, 'cls': cls, 'gnoise': gnoise, 'det': det}[sys.argv0]()
