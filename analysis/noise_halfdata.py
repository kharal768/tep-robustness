"""Noise-scaled budgets and half-data rerun."""
import os
import sys, json
from data_sets import *
from model import FaultClassifier

def stage_noise():
    _, _, (Xh, yh, _) = load_sets(); F = front()
    cal = pd.read_csv('results_r5_calib.csv'); sig = cal[cal.kind == 'coord'].sort_values('coord').noise_sd.values
    rng = np.random.default_rng(0); rows = []
    for tag, hp in (('standard', hp_of(F.iloc[1])), ('adv_pgd10', hp_of(F.iloc[3]))):
        for s in range(5000, 5005):
            m = FaultClassifier(16, 4, hidden_dim=hp.hidden_dim, depth=2); m.load_state_dict(torch.load(f'models/raw002_{tag}_{s}.pt')); m.eval()
            for k in (0.0, 1.0, 2.0):
                pcs = []
                for d in range(10 if k > 0 else 1):
                    Z = Xh + k * sig * rng.standard_normal(Xh.shape)
                    pcs.append(rec(pred(m, torch.as_tensor(Z, dtype=torch.float32)), yh))
                pc = np.mean(pcs, 0)
                rows.append(dict(model=tag, seed=s, noise_mult=k, macro=pc.mean(), pc=json.dumps(np.round(pc, 4).tolist())))
    R = pd.DataFrame(rows); R.to_csv('results_r9_noise.csv', index=False)
    for (m_, k), g in R.groupby(['model', 'noise_mult']):
        print(m_, k, round(g.macro.mean(), 3), np.mean([json.loads(x) for x in g.pc], 0).round(3))

def stage_half():
    (Xtr, ytr, rtr), (Xv, yv, _), (Xh, yh, _) = load_sets(); F = front()
    runs_ = np.unique(rtr); keep = set(np.random.default_rng(1).choice(runs_, size=len(runs_) // 2, replace=False))
    m_ = np.array([r in keep for r in rtr]); Xtr, ytr = Xtr[m_], ytr[m_]
    print('half training', len(ytr), np.bincount(ytr), flush=True); rows = []
    for s in (5000, 5001, 5002):
        h10 = hp_of(F.iloc[3]); h10.pgd_steps_train = 10
        for tag, hp, alpha in (('standard', hp_of(F.iloc[1]), 0.004), ('adv_pgd10', h10, 0.005)):
            c = EvalConfig(eps=0.02, pgd_alpha=alpha, pgd_steps_eval=10)
            m, _ = train_model(hp, Xtr, ytr, Xv, yv, 4, c, epochs=50, seed=s); m.eval()
            ok = strong_ps(m, Xh, yh, 0.02); r = dict(model=tag, seed=s, **summ(m, Xh, yh, 0.02, ok)); rows.append(r)
            pd.DataFrame(rows).to_csv('results_r9_half.csv', index=False); print(tag, s, round(r['clean_macro'], 3), round(r['worst_macro'], 3), r['worst_pc'], flush=True)

if __name__ == '__main__':
    {'noise': stage_noise, 'half': stage_half}[sys.argv[1]]()
