"""APGD-CE/DLR and Square attacks with the product block held fixed."""
import os
import sys; sys.argv = [sys.argv[0], 'cls']
from product_fixed import *
import torchattacks as ta
class W12(torch.nn.Module):
    def __init__(s, f, lo, R, P): super().__init__(); s.f = f; s.lo = lo; s.R = R; s.P = P
    def forward(s, u): return s.f(torch.cat([s.lo + u.reshape(u.shape[0], 12) * s.R, s.P], 1))
def main():
    _, _, (Xh, yh, _) = load_sets(); z = torch.as_tensor(Xh, dtype=torch.float32); yt = torch.as_tensor(yh); ev = torch.as_tensor(EPSV, dtype=torch.float32)
    c = Xh[:, :12].mean(0); dev = np.abs(Xh[:, :12] - c).max(0) + EPSV[:12]; e = float((EPSV[:12] / (2 * dev)).min()) * 0.99
    R = torch.as_tensor(EPSV[:12] / e, dtype=torch.float32); lo = torch.as_tensor(c, dtype=torch.float32) - R / 2; P = z[:, 12:]
    u = ((z[:, :12] - lo) / R).reshape(-1, 12, 1, 1); assert float(u.min()) >= 0 and float(u.max()) <= 1; rows = []
    for pre, seeds in (('raw002', range(5000, 5005)), ('noise2', range(5000, 5003))):
        for tag in ('standard', 'adv_pgd10'):
            for s in seeds:
                f = wrap(load(tag, s, pre), pre); ok = vpgd(f, z, yt, ev); okp = ok.copy(); w = W12(f, lo, R, P).eval()
                for loss in ('ce', 'dlr'):
                    ix = np.where(ok)[0]
                    if len(ix) == 0: break
                    ti = torch.as_tensor(ix); w.P = P[ti]
                    xa = ta.APGD(w, norm='Linf', eps=e, steps=100, loss=loss, n_restarts=1, seed=0)(u[ti], yt[ti])
                    with torch.no_grad(): ok[ix] &= (w(xa).argmax(1) == yt[ti]).numpy()
                wp = np.array([ok[yh == k].mean() for k in range(4)]); wpp = np.array([okp[yh == k].mean() for k in range(4)])
                rows.append(dict(models=pre, model=tag, seed=s, pgd_macro=wpp.mean(), worst_macro=wp.mean(), worst_min=wp.min(), worst_pc=json.dumps(np.round(wp, 4).tolist())))
                pd.DataFrame(rows).to_csv('results_r16_apgd.csv', index=False); print(pre, tag, s, round(wpp.mean(), 3), round(wp.mean(), 3), wp.round(2), flush=True)
main()
