"""run_pipeline.py

Data loading, run-level splitting and the hyperparameter search pipeline:

  raw compositions --ILR--> z --search over (train_model, evaluate_objectives)--> non-dominated set

Data sources:

  --data path/to/file.csv
      Columns x1..xD hold the raw compositional parts (need not be closed)
      and "label" holds the class (0 = normal, 1..K = faults).

  --synthetic-smoke-test
      A synthetic Dirichlet dataset used only to check that the code runs.

Usage:
    python run_pipeline.py --synthetic-smoke-test --n-init 4 --n-iter 4 --epochs 5
    python run_pipeline.py --data my_fault_data.csv --n-init 6 --n-iter 14 --epochs 30
"""

import argparse
import sys

import numpy as np
import pandas as pd
import torch

from coda import ILRTransform
from train import Hyperparams, EvalConfig, train_model
from mobo import run_mobo
import mobo as mobo_module


def make_synthetic_dataset(D: int = 6, n_classes: int = 3, n_per_class: int = 200,
                            seed: int = 0):
    """
    FOR PIPELINE TESTING ONLY -- not a substitute for real process data.
    Class 0 = nominal (symmetric Dirichlet). Classes 1..K-1 = faults,
    modeled as a Dirichlet with a perturbed concentration vector
    (a subset of parts inflated), loosely mimicking a mean-shift fault
    in a subset of compositional parts.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    base_alpha = np.ones(D) * 5.0
    for c in range(n_classes):
        alpha = base_alpha.copy()
        if c > 0:
            shifted_parts = rng.choice(D, size=max(1, D // 3), replace=False)
            alpha[shifted_parts] *= (1.5 + c)
        samples = rng.dirichlet(alpha, size=n_per_class)
        X.append(samples)
        y.append(np.full(n_per_class, c))
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def load_csv_dataset(path: str):
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column (0=nominal, 1..K=fault classes).")
    label_col = df["label"].values.astype(int)
    meta_cols = {"label", "run_id", "t"}
    part_cols = [c for c in df.columns if c not in meta_cols]
    if len(part_cols) < 2:
        raise ValueError("Need at least 2 feature columns besides 'label'/'run_id'/'t'.")
    already_ilr = all(c.startswith("z") for c in part_cols)
    X = df[part_cols].values.astype(float)
    run_id = df["run_id"].values if "run_id" in df.columns else None
    t = df["t"].values if "t" in df.columns else None
    return X, label_col, run_id, t, already_ilr


def split(X, y, run_id=None, t=None, val_frac=0.25, seed=0):
    """
    Splitting strategy depends on what the data supports:

    - If run_id/t are absent (generic CSV, i.i.d. rows assumed): random
      row split.
    - If run_id is present but each label is backed by only one distinct
      run (single-trajectory data): falls back to a time-ordered BLOCK
      split within each run (train = earlier timepoints, val = later
      timepoints), to at least avoid interleaving adjacent, highly
      autocorrelated timepoints across the boundary.
    - If run_id is present AND multiple independent runs exist per label
      (proper multi-run data): performs a LEAVE-RUNS-OUT split -- entire
      simulation runs are held out for validation, so no run's timepoints
      appear in both train and val. This is the methodologically correct
      choice when independent replicates are available, since it tests
      generalization across simulation trajectories rather than just
      across time within one trajectory.
    """
    if run_id is None:
        rng = np.random.default_rng(seed)
        n = len(y)
        idx = rng.permutation(n)
        n_val = int(n * val_frac)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    runs_per_label = {}
    for lab in np.unique(y):
        mask = (y == lab)
        runs_per_label[lab] = np.unique(run_id[mask])

    multi_run = all(len(runs) > 1 for runs in runs_per_label.values())

    if not multi_run:
        return _block_split_within_run(X, y, run_id, t, val_frac)
    return _leave_runs_out_split(X, y, run_id, val_frac, seed)


def _block_split_within_run(X, y, run_id, t, val_frac):
    train_idx, val_idx = [], []
    for run in np.unique(run_id):
        mask = np.where(run_id == run)[0]
        order = mask[np.argsort(t[mask])]
        n = len(order)
        n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
        cutoff = n - n_val
        train_idx.extend(order[:cutoff])
        val_idx.extend(order[cutoff:])
    train_idx, val_idx = np.array(train_idx), np.array(val_idx)
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def _leave_runs_out_split(X, y, run_id, val_frac, seed):
    rng = np.random.default_rng(seed)
    # Hold out val_frac of the runs *contributing to each label*, at the
    # run level -- so an entire run's timepoints go to one side only.
    val_runs = set()
    for lab in np.unique(y):
        mask = (y == lab)
        runs = np.unique(run_id[mask])
        n_val_runs = max(1, int(round(len(runs) * val_frac)))
        chosen = rng.choice(runs, size=n_val_runs, replace=False)
        val_runs.update(chosen.tolist())

    val_mask = np.array([r in val_runs for r in run_id])
    train_mask = ~val_mask
    return X[train_mask], y[train_mask], X[val_mask], y[val_mask]


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=str, help="Path to CSV with compositional parts + 'label' column.")
    src.add_argument("--synthetic-smoke-test", action="store_true",
                      help="Run on synthetic data ONLY to verify the pipeline executes. Not real results.")

    ap.add_argument("--eps", type=float, default=0.3, help="PGD perturbation budget in ILR space.")
    ap.add_argument("--pgd-alpha", type=float, default=0.05, help="PGD step size.")
    ap.add_argument("--pgd-steps-eval", type=int, default=10, help="PGD steps used at evaluation time.")
    ap.add_argument("--epochs", type=int, default=30, help="Training epochs per evaluated configuration.")
    ap.add_argument("--n-init", type=int, default=6, help="Sobol initial design size.")
    ap.add_argument("--n-iter", type=int, default=14, help="MOBO sequential iterations.")
    ap.add_argument("--n-seeds", type=int, default=3,
                     help="Training seeds averaged per hyperparameter configuration. "
                          "Higher = more reliable objective estimate, more compute. "
                          "Use 1 to disable averaging (not recommended -- objectives "
                          "can be highly seed-sensitive).")
    ap.add_argument("--metric", type=str, default="macro", choices=["macro", "min"],
                     help="How g1 (clean accuracy) and g3 (robust accuracy) are "
                          "aggregated across classes. 'macro' (default) averages "
                          "per-class recall. 'min' "
                          "uses the WORST-performing class's recall instead of the "
                          "average -- a stricter metric that directly penalizes "
                          "the partial-collapse exploit documented in train.py's "
                          "docstring (a classifier can score well under 'macro' "
                          "by nailing the majority class and ignoring the rest; "
                          "it cannot do this under 'min', since any ignored class "
                          "drags the score to 0). Use 'min' to search for "
                          "configurations that are genuinely non-degenerate on "
                          "every class, not just on average.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-hidden-dim", type=int, default=None,
                     help="Override the search space's upper bound on hidden layer "
                          "width (default: 128). Set higher (e.g. 384 or 512) to test "
                          "whether additional model capacity helps.")
    ap.add_argument("--out", type=str, default="pareto_front.csv")
    ap.add_argument("--checkpoint", type=str, default=None,
                     help="Path to save/resume MOBO progress. If the file exists, "
                          "resumes from it. Enables running a large search in many "
                          "short chunks. Defaults to <out>.ckpt if not given.")
    args = ap.parse_args()

    if args.max_hidden_dim is not None:
        mobo_module.BOUNDS = mobo_module.BOUNDS.clone()
        mobo_module.BOUNDS[1, 2] = float(args.max_hidden_dim)
        print(f"Search space override: hidden_dim upper bound set to "
              f"{args.max_hidden_dim} (default is 128).")

    if args.synthetic_smoke_test:
        print("=" * 78)
        print("SYNTHETIC SMOKE TEST -- verifying pipeline runs end-to-end.")
        print("This data is NOT real and results here must NOT be reported")
        print("as experimental findings.")
        print("=" * 78)
        X, y = make_synthetic_dataset(seed=args.seed)
        run_id, t = None, None
        D = X.shape[1]
        ilr = ILRTransform(D)
        Z = ilr.forward(X)
    else:
        X, y, run_id, t, already_ilr = load_csv_dataset(args.data)
        if run_id is not None:
            runs_per_label = {lab: len(set(run_id[y == lab])) for lab in np.unique(y)}
            if all(n > 1 for n in runs_per_label.values()):
                print(f"Detected multiple independent runs per label {runs_per_label} "
                      "-> using LEAVE-RUNS-OUT split (entire simulation runs held out "
                      "for validation).")
            else:
                print(f"Only one run per some labels {runs_per_label} -> falling back to "
                      "time-ordered block split within each run.")
        if already_ilr:
            print("Detected pre-ILR-transformed columns (z1..zK) -> using them directly "
                  "as model input, skipping this script's own ILR step. (Use this when "
                  "ILR was applied upstream, e.g. separately per compositional sub-block.)")
            Z = X
        else:
            D = X.shape[1]
            ilr = ILRTransform(D)
            Z = ilr.forward(X)

    n_classes = int(y.max()) + 1
    D = Z.shape[1]

    Ztr, ytr, Zval, yval = split(Z, y, run_id=run_id, t=t, val_frac=0.25, seed=args.seed)
    print(f"Dataset: {len(y)} samples, D={D} parts, {n_classes} classes "
          f"({len(ytr)} train / {len(yval)} val)")

    eval_cfg = EvalConfig(eps=args.eps, pgd_alpha=args.pgd_alpha,
                           pgd_steps_eval=args.pgd_steps_eval, norm="linf")

    def evaluate_fn(hp: Hyperparams):
        """
        Trains --n-seeds independent models under this hyperparameter
        configuration (same data, different weight-init/shuffling seed)
        and returns the MEAN objective vector. This matters: we found
        empirically that a single seed can be highly unreliable here --
        the same config can collapse to majority-class prediction on
        some seeds and train normally on others. Averaging gives MOBO a
        signal that reflects the hyperparameters, not a lucky/unlucky
        initialization. Per-seed spread is printed so collapse-prone
        configs are visible, not just averaged away silently.
        """
        objs = []
        for s in range(args.n_seeds):
            _, o = train_model(hp, Ztr, ytr, Zval, yval,
                                n_classes=n_classes, eval_cfg=eval_cfg,
                                epochs=args.epochs, seed=args.seed * 1000 + s,
                                metric=args.metric)
            objs.append(o)
        objs = np.array(objs)
        mean_obj = tuple(objs.mean(axis=0))
        if args.n_seeds > 1:
            std_obj = objs.std(axis=0)
            print(f"    (n_seeds={args.n_seeds}) clean_acc std={std_obj[0]:.3f} "
                  f"FAR std={std_obj[1]:.3f} robust_acc std={std_obj[2]:.3f}")
        return mean_obj

    result = run_mobo(evaluate_fn, n_init=args.n_init, n_iter=args.n_iter, seed=args.seed,
                       checkpoint_path=args.checkpoint or (args.out + ".ckpt"))

    rows = []
    for i in range(result.X.shape[0]):
        hp = result.hparams[i]
        g1, one_minus_far, g3 = result.Y[i].tolist()
        rows.append({
            "lr": hp.lr, "weight_decay": hp.weight_decay, "hidden_dim": hp.hidden_dim,
            "lambda_adv": hp.lambda_adv, "pgd_steps_train": hp.pgd_steps_train,
            "clean_accuracy": g1, "false_alarm_rate": 1.0 - one_minus_far,
            "robust_accuracy": g3, "on_pareto_front": bool(result.pareto_mask[i]),
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)

    print(f"\nSaved {len(out_df)} evaluated configurations to {args.out}")
    print(f"{int(result.pareto_mask.sum())} configurations on the approximate Pareto front:")
    print(out_df[out_df.on_pareto_front].to_string(index=False))


if __name__ == "__main__":
    main()
