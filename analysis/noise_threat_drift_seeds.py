"""Noise threat model, detection under drift and the ten-seed check."""
import os
import sys; sys.argv0 = sys.argv[1]; sys.argv = [sys.argv[0]]
from detection_eval import *
from model import FaultClassifier
from bounds_centre_lp import solve
SIG = pd.read_csv('results_r5_calib.csv').query("kind=='coord'").sort_values('coord').noise_sd.values
C = 2.0
def noise():
    (Xtr, ytr, _), (Xv, yv, _), (Xh, yh, _) = load_sets(); F = front(); A, B, H = Xtr / SIG, Xv / SIG, Xh / SIG; rows = []
    b = solve(H, yh, C, centres=True); rows.append(dict(model='bound', seed=-1, worst_macro=b['macro'], worst_min=b['worst'], edges=b['edges'])); print(rows[-1], flush=True)
    for s in (5000, 5001, 5002):
        h10 = hp_of(F.iloc[3]); h10.pgd_steps_train = 10
        for tag, hp, alpha in (('standard', hp_of(F.iloc[1]), C / 5), ('adv_pgd10', h10, C / 4)):
            c = EvalConfig(eps=C, pgd_alpha=alpha, pgd_steps_eval=10)
            m, _ = train_model(hp, A, ytr, B, yv, 4, c, epochs=50, seed=s); m.eval(); torch.save(m.state_dict(), f'models/noise2_{tag}_{s}.pt')
            ok = strong_ps(m, H, yh, C); r = dict(model=tag, seed=s, **summ(m, H, yh, C, ok)); rows.append(r)
            pd.DataFrame(rows).to_csv('results_r13_noise.csv', index=False); print(tag, s, round(r['clean_macro'], 3), round(r['worst_macro'], 3), r['worst_pc'], flush=True)
def det():
    X, y, rid, t, _ = load_csv_dataset(D + 'raw_dataset/tep_multirun_composition_data.csv')
    idx = np.arange(len(y), dtype=float)[:, None]; itr, _, iv, _ = split(idx, y, run_id=rid, t=t, val_frac=0.25, seed=0)
    itr = itr[:, 0].astype(int); iv = iv[:, 0].astype(int)
    Atr = X[itr][y[itr] == 0]
    Asm = np.vstack([ewma_t(torch.tensor(Z, dtype=torch.float32)).numpy()[yy == 0] for _, Z, yy, _ in runs(X[itr], y[itr], rid[itr], t[itr])])
    st = PCAStat(Atr); se = PCAStat(Asm); lda = LDA(X[itr], y[itr])
    V = runs(X[iv], y[iv], rid[iv], t[iv]); H = runs(*load_csv_dataset(HOLD)[:4]); F = front(); sg = torch.as_tensor(SIG, dtype=torch.float32)
    dets = {'PCA T2/SPE': lambda z: torch.log(st.index(z)), 'EWMA PCA T2/SPE': lambda z: torch.log(se.index(ewma_t(z))),
            'LDA': lambda z: torch.logsumexp(lda(z)[:, 1:], 1) - lda(z)[:, 0]}
    for tag, hp in (('standard', hp_of(F.iloc[1])), ('adv_pgd10', hp_of(F.iloc[3]))):
        for s in (5000, 5001, 5002):
            m = FaultClassifier(16, 4, hidden_dim=hp.hidden_dim, depth=2); m.load_state_dict(torch.load(f'models/noise2_{tag}_{s}.pt')); m.eval()
            dets[f'net:{tag}:{s}'] = (lambda m: (lambda z: torch.logsumexp(m(z / sg)[:, 1:], 1) - m(z / sg)[:, 0]))(m)
    epsv = C * SIG; rows = []
    for name, fn in dets.items():
        with torch.no_grad(): sv = np.concatenate([fn(torch.tensor(Z, dtype=torch.float32)).numpy()[yy == 0] for _, Z, yy, _ in V])
        thr = float(np.quantile(sv, 0.99))
        for r, Z, yy, tt in H:
            with torch.no_grad(): a0 = (fn(torch.tensor(Z, dtype=torch.float32)) > thr).numpy()
            if (yy > 0).any():
                post = (tt >= ONSET).astype(float); Za = attack_seq_vec(fn, Z, -1, epsv, mask=post)
                Za = torch.where(torch.tensor(post > 0)[:, None], Za, torch.tensor(Z, dtype=torch.float32))
            else: Za = attack_seq_vec(fn, Z, +1, epsv)
            with torch.no_grad(): a1 = (fn(Za) > thr).numpy()
            for cond, a in (('clean', a0), ('attacked', a1)): rows.append(dict(detector=name, run=r, cls=int(yy.max()), cond=cond, **alarm_metrics(a, yy, tt)))
        pd.DataFrame(rows).to_csv('results_r13_det.csv', index=False); print(name, flush=True)
def drift():
    _, _, (Xh, yh, rh) = load_sets(); F = front(); rows = []
    for tag, hp in (('standard', hp_of(F.iloc[1])), ('adv_pgd10', hp_of(F.iloc[3]))):
        for s in range(5000, 5005):
            m = FaultClassifier(16, 4, hidden_dim=hp.hidden_dim, depth=2); m.load_state_dict(torch.load(f'models/raw002_{tag}_{s}.pt')); m.eval()
            for bud, ev in (('raw0.02', 0.02 * np.ones(16)), ('2sigma', C * SIG)):
                ev = torch.as_tensor(ev, dtype=torch.float32); ok = np.zeros(len(yh), bool)
                for r in np.unique(rh):
                    k = rh == r; z = torch.as_tensor(Xh[k], dtype=torch.float32); yt = torch.as_tensor(yh[k]); best = None
                    for rs in range(3):
                        g = torch.Generator().manual_seed(rs); d = (torch.rand(16, generator=g) * 2 - 1) * ev
                        for _ in range(100):
                            d.requires_grad_(True); L = Fn.cross_entropy(m(z + d), yt); gr, = torch.autograd.grad(L, d)
                            d = torch.max(torch.min(d.detach() + ev / 20 * gr.sign(), ev), -ev)
                        with torch.no_grad(): c_ = (m(z + d).argmax(1) == yt).numpy()
                        best = c_ if best is None or c_.sum() < best.sum() else best
                    ok[k] = best
                pc = np.array([ok[yh == c].mean() for c in range(4)])
                rows.append(dict(model=tag, seed=s, budget=bud, macro=pc.mean(), pc=json.dumps(np.round(pc, 4).tolist())))
                pd.DataFrame(rows).to_csv('results_r13_drift.csv', index=False); print(tag, s, bud, round(pc.mean(), 3), pc.round(2), flush=True)
def seeds():
    (Xtr, ytr, _), (Xv, yv, _), (Xh, yh, _) = load_sets(); F = front(); rows = []
    for s in range(5005, 5010):
        for tag, hp, alpha in (('standard', hp_of(F.iloc[1]), 0.004), ('adv_pgd10', hp_of(F.iloc[3]), 0.005)):
            c = EvalConfig(eps=0.02, pgd_alpha=alpha, pgd_steps_eval=10)
            m, _ = train_model(hp, Xtr, ytr, Xv, yv, 4, c, epochs=50, seed=s); m.eval()
            ok = strong_ps(m, Xh, yh, 0.02); r = dict(model=tag, seed=s, **summ(m, Xh, yh, 0.02, ok)); rows.append(r)
            pd.DataFrame(rows).to_csv('results_r13_seeds.csv', index=False); print(tag, s, round(r['worst_macro'], 3), r['worst_pc'], flush=True)
if __name__ == '__main__': {'noise': noise, 'det': det, 'drift': drift, 'seeds': seeds}[sys.argv0]()
