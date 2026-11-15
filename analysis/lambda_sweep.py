"""Sweep of the regularisation weight lambda."""
import os
import sys
from data_sets import *
def main():
    (Xtr, ytr, _), (Xv, yv, _), (Xh, yh, _) = load_sets(); F = front(); rows = []
    for lam in (0.1, 0.3, 0.55, 0.9):
        for s in (5000, 5001, 5002):
            hp = hp_of(F.iloc[3]); hp.pgd_steps_train = 10; hp.lambda_adv = lam
            c = EvalConfig(eps=0.02, pgd_alpha=0.005, pgd_steps_eval=10)
            m, _ = train_model(hp, Xtr, ytr, Xv, yv, 4, c, epochs=50, seed=s); m.eval()
            ok = strong_ps(m, Xh, yh, 0.02); r = dict(lam=lam, seed=s, **summ(m, Xh, yh, 0.02, ok)); rows.append(r)
            pd.DataFrame(rows).to_csv('results_r10_lambda.csv', index=False)
            print(lam, s, round(r['clean_macro'], 3), round(r['worst_macro'], 3), r['worst_pc'], flush=True)
if __name__ == '__main__': main()
