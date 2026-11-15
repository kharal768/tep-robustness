"""
run_mobo_resumable.py

Same MOBO procedure as mobo.run_mobo, but checkpoints every evaluated
configuration to a CSV cache file as it goes, and picks up where it left
off on the next invocation. A long search can therefore be resumed without retraining finished
configurations.

Usage: just re-run this script repeatedly with the same --out path;
it stops early (soft time budget) and picks back up next time, until
TARGET_N_INIT + TARGET_N_ITER total configurations have been evaluated.
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch

from coda import ILRTransform
from train import Hyperparams, EvalConfig, train_model
from mobo import (
    BOUNDS, REF_POINT, decode_x, to_botorch_objectives, _fit_model, _pareto_mask,
)
from run_pipeline import load_csv_dataset, make_synthetic_dataset, split
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize


def load_cache(path):
    try:
        df = pd.read_csv(path)
        X = torch.tensor(df[["x0", "x1", "x2", "x3", "x4"]].values, dtype=torch.double)
        Y = torch.tensor(df[["y0", "y1", "y2"]].values, dtype=torch.double)
        return df, X, Y
    except FileNotFoundError:
        return None, None, None


def save_cache(path, X, Y, hparams):
    rows = []
    for i in range(X.shape[0]):
        hp = hparams[i]
        rows.append({
            "x0": X[i, 0].item(), "x1": X[i, 1].item(), "x2": X[i, 2].item(),
            "x3": X[i, 3].item(), "x4": X[i, 4].item(),
            "y0": Y[i, 0].item(), "y1": Y[i, 1].item(), "y2": Y[i, 2].item(),
            "lr": hp.lr, "weight_decay": hp.weight_decay, "hidden_dim": hp.hidden_dim,
            "lambda_adv": hp.lambda_adv, "pgd_steps_train": hp.pgd_steps_train,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", default="scaleup_cache.csv")
    ap.add_argument("--target-n-init", type=int, default=8)
    ap.add_argument("--target-n-iter", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--pgd-alpha", type=float, default=0.03)
    ap.add_argument("--pgd-steps-eval", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-budget-sec", type=float, default=230)
    args = ap.parse_args()

    t_start = time.time()

    X_full, y_full, run_id, t_col, already_ilr = load_csv_dataset(args.data)
    if already_ilr:
        Z = X_full
    else:
        ilr = ILRTransform(X_full.shape[1])
        Z = ilr.forward(X_full)
    n_classes = int(y_full.max()) + 1
    Ztr, ytr, Zval, yval = split(Z, y_full, run_id=run_id, t=t_col, val_frac=0.25, seed=args.seed)

    eval_cfg = EvalConfig(eps=args.eps, pgd_alpha=args.pgd_alpha,
                           pgd_steps_eval=args.pgd_steps_eval, norm="linf")

    def evaluate_fn(hp: Hyperparams):
        _, objectives = train_model(hp, Ztr, ytr, Zval, yval, n_classes=n_classes,
                                     eval_cfg=eval_cfg, epochs=args.epochs, seed=args.seed)
        return objectives

    _, X, Y = load_cache(args.cache)
    hparams = []
    if X is not None:
        df_cache = pd.read_csv(args.cache)
        for _, row in df_cache.iterrows():
            hparams.append(Hyperparams(lr=row.lr, weight_decay=row.weight_decay,
                                        hidden_dim=int(row.hidden_dim), depth=2,
                                        lambda_adv=row.lambda_adv,
                                        pgd_steps_train=int(row.pgd_steps_train)))
    n_done = 0 if X is None else X.shape[0]
    print(f"Resuming from cache: {n_done} configs already evaluated.")

    torch.manual_seed(args.seed)
    X_init_full = draw_sobol_samples(bounds=BOUNDS, n=args.target_n_init, q=1,
                                      seed=args.seed).squeeze(1).double()

    def time_left():
        return args.time_budget_sec - (time.time() - t_start)

    # --- finish initial design if needed ---
    while n_done < args.target_n_init and time_left() > 15:
        hp = decode_x(X_init_full[n_done])
        g1, g2, g3 = evaluate_fn(hp)
        y = torch.tensor([to_botorch_objectives(g1, g2, g3)], dtype=torch.double)
        X = X_init_full[n_done:n_done+1] if X is None else torch.cat([X, X_init_full[n_done:n_done+1]])
        Y = y if Y is None else torch.cat([Y, y])
        hparams.append(hp)
        n_done += 1
        save_cache(args.cache, X, Y, hparams)
        print(f"[init {n_done}/{args.target_n_init}] {hp} -> "
              f"acc={g1:.3f} FAR={g2:.3f} robust_acc={g3:.3f} "
              f"(elapsed {time.time()-t_start:.0f}s)")

    # --- BO iterations ---
    n_bo_done = max(0, n_done - args.target_n_init)
    while n_bo_done < args.target_n_iter and n_done >= args.target_n_init and time_left() > 20:
        model = _fit_model(X, Y)
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model, ref_point=REF_POINT.tolist(),
            X_baseline=normalize(X, BOUNDS), prune_baseline=True,
        )
        norm_bounds = torch.stack([torch.zeros(5, dtype=torch.double),
                                    torch.ones(5, dtype=torch.double)])
        candidate_norm, _ = optimize_acqf(acq_function=acqf, bounds=norm_bounds, q=1,
                                           num_restarts=5, raw_samples=64)
        candidate = unnormalize(candidate_norm, BOUNDS)
        hp = decode_x(candidate[0])
        g1, g2, g3 = evaluate_fn(hp)
        y = torch.tensor([to_botorch_objectives(g1, g2, g3)], dtype=torch.double)
        X = torch.cat([X, candidate])
        Y = torch.cat([Y, y])
        hparams.append(hp)
        n_done += 1
        n_bo_done += 1
        save_cache(args.cache, X, Y, hparams)
        print(f"[bo-iter {n_bo_done}/{args.target_n_iter}] {hp} -> "
              f"acc={g1:.3f} FAR={g2:.3f} robust_acc={g3:.3f} "
              f"(elapsed {time.time()-t_start:.0f}s)")

    total_target = args.target_n_init + args.target_n_iter
    if n_done >= total_target:
        mask = _pareto_mask(Y)
        print(f"\nDONE: {n_done}/{total_target} configs evaluated. "
              f"{int(mask.sum())} on Pareto front.")
    else:
        print(f"\nPaused at {n_done}/{total_target} configs (time budget reached). "
              f"Re-run the same command to resume from {args.cache}.")


if __name__ == "__main__":
    main()
