"""Attacks restricted to subsets of the composition blocks."""
import os
import sys; sys.argv = [sys.argv[0]]
from detection_eval import *
def main():
    X, y, rid, t, _ = load_csv_dataset(D + 'raw_dataset/tep_multirun_composition_data.csv')
    idx = np.arange(len(y), dtype=float)[:, None]; itr, _, iv, _ = split(idx, y, run_id=rid, t=t, val_frac=0.25, seed=0)
    itr = itr[:, 0].astype(int); iv = iv[:, 0].astype(int)
    (Xtr_a, _, _), _, _ = load_sets(); sd = Xtr_a.std(0)
    Atr = X[itr][y[itr] == 0]
    Asm = np.vstack([ewma_t(torch.tensor(Z, dtype=torch.float32)).numpy()[yy == 0] for _, Z, yy, _ in runs(X[itr], y[itr], rid[itr], t[itr])])
    st = PCAStat(Atr); se = PCAStat(Asm)
    V = runs(X[iv], y[iv], rid[iv], t[iv]); H = runs(*load_csv_dataset(HOLD)[:4])
    dets = {'PCA T2/SPE': lambda z: torch.log(st.index(z)), 'EWMA PCA T2/SPE': lambda z: torch.log(se.index(ewma_t(z)))}
    prod = np.zeros(16); prod[12:] = 1; rows = []
    for name, fn in dets.items():
        with torch.no_grad():
            sv = np.concatenate([fn(torch.tensor(Z, dtype=torch.float32)).numpy()[yy == 0] for _, Z, yy, _ in V])
        thr = float(np.quantile(sv, 0.99))
        for sub, msk in (('all', np.ones(16)), ('product', prod), ('other', 1 - prod)):
            epsv = 0.25 * sd * msk
            for r, Z, yy, tt in H:
                if (yy > 0).any():
                    post = (tt >= ONSET).astype(float); Za = attack_seq_vec(fn, Z, -1, epsv, mask=post)
                    Za = torch.where(torch.tensor(post > 0)[:, None], Za, torch.tensor(Z, dtype=torch.float32))
                else:
                    Za = attack_seq_vec(fn, Z, +1, epsv)
                with torch.no_grad(): a1 = (fn(Za) > thr).numpy()
                rows.append(dict(detector=name, coords=sub, run=r, cls=int(yy.max()), **alarm_metrics(a1, yy, tt)))
            pd.DataFrame(rows).to_csv('results_r11_mech.csv', index=False); print(name, sub, flush=True)
if __name__ == '__main__': main()
