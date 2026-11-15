"""Sensitivity of results to the calibration window and detection threshold."""
import os
import sys, json
import data_sets
from data_sets import *
from detection_eval import attack_seq_vec
from detection_cnn import runs, ewma_t, PCAStat, alarm_metrics
from model import FaultClassifier

def stage_win():
    rows = []; F = front()
    for excl_min in (30, 120):
        data_sets.EXCL = excl_min // 3
        (Xtr, ytr_, _), (Xv, yv_, _), (Xh, yh, _) = data_sets.load_sets()
        print('window', excl_min, len(ytr_), len(yv_), len(yh), flush=True)
        for s in (5000, 5001, 5002):
            specs = [('standard', hp_of(F.iloc[1]), 0.02 / 5)]
            h10 = hp_of(F.iloc[3]); h10.pgd_steps_train = 10; specs.append(('adv_pgd10', h10, 0.02 / 4))
            for tag, hp, alpha in specs:
                c = EvalConfig(eps=0.02, pgd_alpha=alpha, pgd_steps_eval=10)
                m, _ = train_model(hp, Xtr, ytr_, Xv, yv_, 4, c, epochs=50, seed=s); m.eval()
                ok = strong_ps(m, Xh, yh, 0.02)
                r = dict(window_min=excl_min, model=tag, seed=s, **summ(m, Xh, yh, 0.02, ok)); rows.append(r)
                pd.DataFrame(rows).to_csv('results_r8_window.csv', index=False)
                print(excl_min, tag, s, round(r['clean_macro'], 3), round(r['worst_macro'], 3), r['worst_pc'], flush=True)

def stage_thr():
    X, y, rid, t, _ = load_csv_dataset(D + 'raw_dataset/tep_multirun_composition_data.csv')
    idx = np.arange(len(y), dtype=float)[:, None]; itr, _, iv, _ = split(idx, y, run_id=rid, t=t, val_frac=0.25, seed=0)
    itr = itr[:, 0].astype(int); iv = iv[:, 0].astype(int)
    Atr = X[itr][y[itr] == 0]
    Asm = np.vstack([ewma_t(torch.tensor(Z, dtype=torch.float32)).numpy()[yy == 0] for _, Z, yy, _ in runs(X[itr], y[itr], rid[itr], t[itr])])
    st = PCAStat(Atr); se = PCAStat(Asm)
    V = runs(X[iv], y[iv], rid[iv], t[iv]); H = runs(*load_csv_dataset(HOLD)[:4]); F = front()
    dets = {'PCA T2/SPE': lambda z: torch.log(st.index(z)), 'EWMA PCA T2/SPE': lambda z: torch.log(se.index(ewma_t(z)))}
    for tag, hp in (('standard', hp_of(F.iloc[1])), ('adv_pgd10', hp_of(F.iloc[3]))):
        for s in range(5000, 5005):
            m = FaultClassifier(16, 4, hidden_dim=hp.hidden_dim, depth=2); m.load_state_dict(torch.load(f'models/raw002_{tag}_{s}.pt')); m.eval()
            dets[f'net:{tag}:{s}'] = (lambda m: (lambda z: torch.logsumexp(m(z)[:, 1:], 1) - m(z)[:, 0]))(m)
    rows = []; epsv = 0.02 * np.ones(16)
    for name, fn in dets.items():
        with torch.no_grad():
            sv = np.concatenate([fn(torch.tensor(Z, dtype=torch.float32)).numpy()[yy == 0] for _, Z, yy, _ in V])
        thrs = {q: float(np.quantile(sv, 1 - q)) for q in (0.005, 0.01, 0.05)}
        for r, Z, yy, tt in H:
            with torch.no_grad(): s0 = fn(torch.tensor(Z, dtype=torch.float32)).numpy()
            if (yy > 0).any():
                post = (tt >= ONSET).astype(float); Za = attack_seq_vec(fn, Z, -1, epsv, mask=post)
                Za = torch.where(torch.tensor(post > 0)[:, None], Za, torch.tensor(Z, dtype=torch.float32))
            else:
                Za = attack_seq_vec(fn, Z, +1, epsv)
            with torch.no_grad(): s1 = fn(Za).numpy()
            for q, thr in thrs.items():
                for cond, sc in (('clean', s0), ('attacked', s1)):
                    rows.append(dict(detector=name, far=q, run=r, cls=int(yy.max()), cond=cond, **alarm_metrics(sc > thr, yy, tt)))
        pd.DataFrame(rows).to_csv('results_r8_threshold.csv', index=False); print(name, flush=True)

if __name__ == '__main__':
    {'win': stage_win, 'thr': stage_thr}[sys.argv[1]]()
