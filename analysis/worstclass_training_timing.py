"""Worst-class (group-DRO) training and bound computation timing."""
import os
import sys, time; mode = sys.argv[1]; sys.argv = [sys.argv[0]]
from data_sets import *
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','core'))
from attacks import pgd_attack
from model import FaultClassifier
import torch.nn.functional as Fn
def train_dro(hp, Z, Y, eps, alpha, seed, epochs=50, eta=0.1):
    torch.manual_seed(seed); m = FaultClassifier(16, 4, hidden_dim=hp.hidden_dim, depth=2)
    opt = torch.optim.Adam(m.parameters(), lr=hp.lr, weight_decay=hp.weight_decay); q = torch.ones(4) / 4
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.as_tensor(Z, dtype=torch.float32), torch.as_tensor(Y)), batch_size=64, shuffle=True)
    for _ in range(epochs):
        for zb, yb in dl:
            opt.zero_grad(); lc = Fn.cross_entropy(m(zb), yb)
            za = pgd_attack(m, zb, yb, eps=eps, alpha=alpha, steps=10, norm='linf'); la = Fn.cross_entropy(m(za), yb, reduction='none')
            gl = torch.stack([la[yb == c].mean() if (yb == c).any() else torch.tensor(0.) for c in range(4)])
            with torch.no_grad(): q = q * torch.exp(eta * gl); q = q / q.sum()
            loss = lc + hp.lambda_adv * 4 * (q * gl).sum(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval(); return m
if mode == 'dro':
    (Xtr, ytr, _), _, (Xh, yh, _) = load_sets(); F = front(); rows = []
    for s in range(5000, 5005):
        hp = hp_of(F.iloc[3]); m = train_dro(hp, Xtr, ytr, 0.02, 0.005, s)
        ok = strong_ps(m, Xh, yh, 0.02); r = dict(model='worstclass_dro', seed=s, **summ(m, Xh, yh, 0.02, ok)); rows.append(r)
        pd.DataFrame(rows).to_csv('results_r18_dro.csv', index=False); print(s, round(r['clean_macro'], 3), round(r['worst_macro'], 3), r['worst_pc'], flush=True)
else:
    from bounds_centre_lp import solve
    from dai_comparison import dai_lp
    _, (Xv, yv, _), (Xh, yh, _) = load_sets(); rows = []
    for setn, Z, yy in (('heldout', Xh, yh), ('validation', Xv, yv)):
        t0 = time.time(); r = solve(Z, yy, 0.02, centres=True); t1 = time.time(); d = dai_lp(Z, yy, 0.02); t2 = time.time()
        rows.append(dict(set=setn, centre_macro=r['macro'], centre_sec=t1 - t0, dai_macro=d['dai_macro'], dai_sec=t2 - t1, cliques=d['cliques'])); print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv('results_r18_timing.csv', index=False)
