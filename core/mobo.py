"""
mobo.py

Multiobjective Bayesian optimization over classifier/training
hyperparameters, using Gaussian process surrogates and expected
hypervolume improvement (qNEHVI). It selected the epsilon = 0.15 configurations
evaluated in the analysis scripts.

Search space (5 continuous dims, mapped to Hyperparams):
  0: log10(lr)            in [-4, -1.5]
  1: log10(weight_decay)  in [-6, -2]     (weight_decay=0 handled as a floor)
  2: hidden_dim           in [16, 128]    (rounded to int)
  3: lambda_adv           in [0, 2]
  4: pgd_steps_train      in [0, 10]      (rounded to int)

depth is fixed at 2.

Objectives passed to BoTorch (all MAXIMIZED, BoTorch convention):
  y0 = clean accuracy            (g1, maximize directly)
  y1 = 1 - false_alarm_rate      (g2, minimize false alarms -> maximize 1-FAR)
  y2 = robust accuracy           (g3, maximize directly)
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import SumMarginalLogLikelihood
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize

from train import Hyperparams

BOUNDS = torch.tensor([
    [-4.0, -6.0, 16.0, 0.0, 0.0],
    [-2.3, -2.0, 128.0, 2.0, 10.0],
], dtype=torch.double)

REF_POINT = torch.tensor([0.0, 0.0, 0.0], dtype=torch.double)  # worst-case for each (maximized) objective


def decode_x(x_row: torch.Tensor) -> Hyperparams:
    """Map one raw search-space point (5,) to a Hyperparams config."""
    lr = 10 ** x_row[0].item()
    wd = 10 ** x_row[1].item()
    hidden_dim = int(round(x_row[2].item()))
    lambda_adv = float(x_row[3].item())
    pgd_steps_train = int(round(x_row[4].item()))
    return Hyperparams(lr=lr, weight_decay=wd, hidden_dim=hidden_dim,
                        depth=2, lambda_adv=lambda_adv,
                        pgd_steps_train=pgd_steps_train)


def to_botorch_objectives(g1_acc: float, g2_far: float, g3_robust_acc: float) -> Tuple[float, float, float]:
    return (g1_acc, 1.0 - g2_far, g3_robust_acc)


import json
from pathlib import Path

@dataclass
class MOBOResult:
    X: torch.Tensor            # (n, 5) raw hyperparameter search-space points evaluated
    Y: torch.Tensor            # (n, 3) objectives (all maximized: acc, 1-FAR, robust_acc)
    hparams: list              # list[Hyperparams], same order as X/Y
    pareto_mask: torch.Tensor  # (n,) bool, True for non-dominated points


def _save_checkpoint(path, X, Y, hparams):
    torch.save({
        "X": X, "Y": Y,
        "hparams": [vars(hp) for hp in hparams],
    }, path)


def _load_checkpoint(path):
    ckpt = torch.load(path, weights_only=False)
    hparams = [Hyperparams(**d) for d in ckpt["hparams"]]
    return ckpt["X"], ckpt["Y"], hparams


def _fit_model(X: torch.Tensor, Y: torch.Tensor) -> ModelListGP:
    X_norm = normalize(X, BOUNDS)
    models = [SingleTaskGP(X_norm, Y[:, i:i + 1]) for i in range(Y.shape[-1])]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def _pareto_mask(Y: torch.Tensor) -> torch.Tensor:
    n = Y.shape[0]
    mask = torch.ones(n, dtype=torch.bool)
    for i in range(n):
        for j in range(n):
            if i != j and torch.all(Y[j] >= Y[i]) and torch.any(Y[j] > Y[i]):
                mask[i] = False
                break
    return mask


def run_mobo(evaluate_fn: Callable[[Hyperparams], Tuple[float, float, float]],
             n_init: int = 6, n_iter: int = 14, q: int = 1,
             seed: int = 0, verbose: bool = True,
             checkpoint_path: str = None) -> MOBOResult:
    """
    evaluate_fn: given a Hyperparams config, trains a model and returns
                 (clean_acc, false_alarm_rate, robust_acc).

    checkpoint_path: if given, saves progress after every single
                 evaluation and resumes from it automatically if it
                 already exists. Lets a large search run in many short
                 chunks instead of one long continuous process.
    """
    torch.manual_seed(seed)

    X_list, Y_list, hp_list = [], [], []
    if checkpoint_path and Path(checkpoint_path).exists():
        X_ckpt, Y_ckpt, hp_ckpt = _load_checkpoint(checkpoint_path)
        X_list = [X_ckpt[i] for i in range(X_ckpt.shape[0])]
        Y_list = [Y_ckpt[i] for i in range(Y_ckpt.shape[0])]
        hp_list = hp_ckpt
        if verbose:
            print(f"[resume] loaded {len(X_list)} previously evaluated configs "
                  f"from {checkpoint_path}")

    n_done = len(X_list)

    # --- initial Sobol design (resume-aware) ---
    X_init = draw_sobol_samples(bounds=BOUNDS, n=n_init, q=1, seed=seed).squeeze(1).double()

    for i in range(n_init):
        if n_done > i:
            continue
        hp = decode_x(X_init[i])
        g1, g2, g3 = evaluate_fn(hp)
        y = to_botorch_objectives(g1, g2, g3)
        X_list.append(X_init[i])
        Y_list.append(torch.tensor(y, dtype=torch.double))
        hp_list.append(hp)
        if verbose:
            print(f"[init {i+1}/{n_init}] {hp} -> acc={g1:.3f} FAR={g2:.3f} robust_acc={g3:.3f}")
        if checkpoint_path:
            _save_checkpoint(checkpoint_path, torch.stack(X_list), torch.stack(Y_list), hp_list)

    X = torch.stack(X_list)
    Y = torch.stack(Y_list)

    # --- BO loop (resume-aware) ---
    n_iter_done = max(0, len(X_list) - n_init)
    for it in range(n_iter_done, n_iter):
        model = _fit_model(X, Y)
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=REF_POINT.tolist(),
            X_baseline=normalize(X, BOUNDS),
            prune_baseline=True,
        )
        norm_bounds = torch.stack([torch.zeros(5, dtype=torch.double),
                                    torch.ones(5, dtype=torch.double)])
        candidate_norm, _ = optimize_acqf(
            acq_function=acqf, bounds=norm_bounds, q=q,
            num_restarts=5, raw_samples=64,
        )
        candidate = unnormalize(candidate_norm, BOUNDS)

        for i in range(candidate.shape[0]):
            hp = decode_x(candidate[i])
            g1, g2, g3 = evaluate_fn(hp)
            y = to_botorch_objectives(g1, g2, g3)
            X = torch.cat([X, candidate[i:i + 1]], dim=0)
            Y = torch.cat([Y, torch.tensor([y], dtype=torch.double)], dim=0)
            hp_list.append(hp)
            if verbose:
                print(f"[iter {it+1}/{n_iter}] {hp} -> acc={g1:.3f} FAR={g2:.3f} robust_acc={g3:.3f}")
            if checkpoint_path:
                _save_checkpoint(checkpoint_path, X, Y, hp_list)

    mask = _pareto_mask(Y)
    return MOBOResult(X=X, Y=Y, hparams=hp_list, pareto_mask=mask)
