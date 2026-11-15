"""Published budgets in the composition space and attacks on a sequence-level CNN."""
import os
import sys; sys.argv0 = sys.argv[1]; sys.argv = [sys.argv[0]]
from detection_eval import *
from calibration_bounds import inv, BL, load_all
from bounds_centre_lp import solve
def pub():
    (Xtr, _, _), (Xv, yv, _), (Xh, yh, _) = load_sets(); sd = Xtr.std(0)
    X, y, rid, t, itr, iv = load_all(); nom = np.array([str(r).startswith('label0') for r in rid]) & np.isin(np.arange(len(y)), itr)
    P = pd.read_csv('results_r5_calib.csv').query("kind=='part'"); zbar = X[nom].mean(0); rows = []
    for a, b, name, parts in BL:
        zb = zbar[a:b]; x0 = inv(zb)
        for i, p in enumerate(parts):
            xn = float(P[(P.block == name) & (P.part == p)].noise_share.iloc[0])
            g = np.array([(inv(zb + 1e-6 * np.eye(len(zb))[k])[i] - x0[i]) / 1e-6 for k in range(len(zb))])
            for kk in (0.1, 0.3):
                ev = kk * sd[a:b]; best = max(abs(inv(zb + s * ev * np.sign(g))[i] - x0[i]) for s in (1, -1))
                rows.append(dict(kind='calib', block=name, part=p, share=x0[i], sd=kk, change=best, ratio=best / xn))
    for setn, Z, yy in (('heldout', Xh, yh), ('validation', Xv, yv)):
        r0 = solve(Z / sd, yy, 0.3, centres=False); r1 = solve(Z / sd, yy, 0.3, centres=True)
        row = dict(kind='bound', set=setn, sd=0.3, pair_macro=r0['macro'], centre_macro=r1['macro'], centre_worst=r1['worst'], edges=r1['edges'])
        if r1['edges'] < 200000:
            ri = solve(Z / sd, yy, 0.3, centres=True, integer=True, tl=1200); row['ip_macro'] = ri['macro']
        rows.append(row); print(row, flush=True)
    R = pd.DataFrame(rows); R.to_csv('results_r14_pub.csv', index=False); print(R[(R.kind == 'calib') & (R.share > 0.05)].round(3).to_string())
def windows_idx(X, y, rid, t, w=10):
    out = []
    for r in np.unique(rid):
        k = np.where(rid == r)[0]; k = k[np.argsort(t[k])]; ends = []; labs = []
        for j in range(w - 1, len(k)):
            lab = y[k[j]]; tt = t[k[j]]
            if np.array_equal(X[k[j]], X[k[j - 1]]): continue
            if lab > 0 and ONSET <= tt < ONSET + EXCL: continue
            if lab > 0 and t[k[j - w + 1]] < ONSET: continue
            ends.append(j); labs.append(lab)
        out.append((r, X[k], np.array(ends), np.array(labs)))
    return out
def seq():
    X, y, rid, t, _ = load_csv_dataset(D + 'raw_dataset/tep_multirun_composition_data.csv')
    idx = np.arange(len(y), dtype=float)[:, None]; itr, _, iv, _ = split(idx, y, run_id=rid, t=t, val_frac=0.25, seed=0)
    tr_runs = set(rid[itr[:, 0].astype(int)]); mtr = np.isin(rid, list(tr_runs))
    Wtr, Ytr, _ = windows_f(X[mtr], y[mtr], rid[mtr], t[mtr]); Xh, yh, rh, th, _ = load_csv_dataset(HOLD); RS = windows_idx(Xh, yh, rh, th)
    eps = 0.02; rows = []; ar = torch.arange(10)
    for s in (5000, 5001, 5002):
        for tag, lam in (('standard', 0.0), ('adversarial', 0.55)):
            torch.manual_seed(s); m = CNN(); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(Wtr), torch.tensor(Ytr)), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(s))
            for ep in range(50):
                m.train()
                for xb, yb in dl:
                    loss = Fn.cross_entropy(m(xb), yb)
                    if lam > 0: loss = loss + lam * Fn.cross_entropy(m(pgd_w(m, xb, yb, eps, eps / 5, 4)), yb)
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            m.eval(); okall = []; yall = []
            for r, Z, ends, labs in RS:
                if len(ends) == 0: continue
                z0 = torch.tensor(Z, dtype=torch.float32); ii = torch.as_tensor(ends)[:, None] - 9 + ar[None, :]; yt = torch.as_tensor(labs)
                f = lambda zz: m(zz[ii].permute(0, 2, 1)); ok = (f(z0).argmax(1) == yt).numpy()
                for rs in range(3):
                    g = torch.Generator().manual_seed(rs); d = (torch.rand(z0.shape, generator=g) * 2 - 1) * eps
                    for _ in range(100):
                        d.requires_grad_(True); L = Fn.cross_entropy(f(z0 + d), yt, reduction='sum'); gr, = torch.autograd.grad(L, d)
                        d = (d.detach() + eps / 20 * gr.sign()).clamp(-eps, eps)
                    with torch.no_grad(): ok &= (f(z0 + d).argmax(1) == yt).numpy()
                okall.append(ok); yall.append(labs)
            ok = np.concatenate(okall); Y = np.concatenate(yall); wp = np.array([ok[Y == c].mean() for c in range(4)])
            rows.append(dict(model=tag, seed=s, worst_macro=wp.mean(), worst_pc=json.dumps(np.round(wp, 4).tolist())))
            pd.DataFrame(rows).to_csv('results_r14_seq.csv', index=False); print(tag, s, round(wp.mean(), 3), wp.round(3), flush=True)
if __name__ == '__main__': {'pub': pub, 'seq': seq}[sys.argv0]()
