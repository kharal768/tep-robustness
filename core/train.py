"""train.py

Trains a FaultClassifier with the combined clean + adversarial loss
L(theta; lambda) = L_clean(theta) + lambda * L_adv(theta), where L_adv uses
PGD perturbations in ILR coordinates, and evaluates three objectives:

  g1: clean accuracy (macro by default)
  g2: false-alarm rate (P[predict fault | true label = normal])
  g3: robust accuracy under PGD at budget eps

Label 0 is the normal class throughout.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import FaultClassifier
from attacks import pgd_attack

NOMINAL_CLASS = 0


@dataclass
class Hyperparams:
    lr: float
    weight_decay: float
    hidden_dim: int
    depth: int
    lambda_adv: float          # weight on adversarial loss term
    pgd_steps_train: int       # PGD steps used to build adversarial examples during training


@dataclass
class EvalConfig:
    eps: float                 # perturbation budget (ILR space)
    pgd_alpha: float           # PGD step size
    pgd_steps_eval: int        # PGD steps used at evaluation time (can differ from training)
    norm: str = "linf"


def train_model(hp: Hyperparams, train_z, train_y, val_z, val_y,
                 n_classes: int, eval_cfg: EvalConfig,
                 epochs: int = 30, batch_size: int = 64, device: str = "cpu",
                 seed: int = 0, metric: str = "macro"):
    """
    Trains one classifier under a fixed hyperparameter configuration and
    returns (model, objectives) where objectives = (g1_acc, g2_far, g3_robust_acc).

    g1 and g3 are returned as accuracies (higher is better);
    g2 is returned as a rate (lower is better). Sign conventions for the
    MOBO wrapper (maximize vs minimize) are handled in mobo.py, not here.

    metric: "macro" (default) or "min"
            (stricter -- minimum per-class recall rather than the
            average; see train.py module docstring / _min_class_accuracy
            for why this matters and what exploit it closes). Return
            signature is unchanged either way, so this is a drop-in
            replacement for any existing caller -- only which numbers
            end up in g1/g3 changes.
    """
    torch.manual_seed(seed)
    device = torch.device(device)

    in_dim = train_z.shape[1]
    model = FaultClassifier(in_dim, n_classes, hidden_dim=hp.hidden_dim,
                             depth=hp.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr,
                            weight_decay=hp.weight_decay)

    train_ds = TensorDataset(torch.as_tensor(train_z, dtype=torch.float32),
                              torch.as_tensor(train_y, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for zb, yb in train_loader:
            zb, yb = zb.to(device), yb.to(device)
            opt.zero_grad()

            logits_clean = model(zb)
            loss_clean = F.cross_entropy(logits_clean, yb)

            if hp.lambda_adv > 0 and hp.pgd_steps_train > 0:
                zb_adv = pgd_attack(model, zb, yb, eps=eval_cfg.eps,
                                     alpha=eval_cfg.pgd_alpha,
                                     steps=hp.pgd_steps_train,
                                     norm=eval_cfg.norm)
                logits_adv = model(zb_adv)
                loss_adv = F.cross_entropy(logits_adv, yb)
            else:
                loss_adv = torch.tensor(0.0, device=device)

            loss = loss_clean + hp.lambda_adv * loss_adv
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

    g1, g2, g3, diagnostics = evaluate_objectives(model, val_z, val_y, eval_cfg,
                                                   device=device, metric=metric)
    return model, (g1, g2, g3)


@torch.no_grad()
def _accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


@torch.no_grad()
def _macro_accuracy(logits, y, n_classes):
    """
    Macro-averaged (per-class) recall, i.e. mean of each class's own
    accuracy, rather than overall/micro accuracy.

    This matters a lot here: under class imbalance (our nominal class
    is the majority), a degenerate classifier that always predicts the
    majority class scores HIGH on overall accuracy, and -- worse --
    scores artificially high on "robust accuracy" too, since a constant
    function is trivially invariant to input perturbation (it ignores
    the input entirely). That means overall-accuracy objectives can
    make a completely useless, input-ignoring classifier look like the
    most "robust" option to a multiobjective search, which is exactly
    what we observed in practice ("the
    degenerate-classifier exploit"). Macro accuracy fixes this: a
    constant predictor only gets credit for the one class it always
    predicts (~1/n_classes), not for the classes it silently ignores.

    KNOWN RESIDUAL EXPLOIT (confusion-matrix analysis): macro-averaging prevents *complete* collapse
    but not *partial* collapse. A classifier that gets one class (e.g.
    the majority Nominal class) almost perfectly and every other class
    at or near zero can still clear a "> chance level" macro-accuracy
    filter, because near-1.0 on one class plus a small non-zero
    contribution from a second class is arithmetically enough. This was
    observed directly: the highest-"robust-accuracy" configuration in an
    early search had 0.999/0.091/0.000/0.000 per-class recall (clean)
    and 0.961/0.000/0.000/0.000 (robust) -- a Nominal specialist, not a
    robust fault diagnoser, despite a macro accuracy of 0.269 (> the
    0.25 chance threshold used to filter degenerate configs). See
    _min_class_accuracy below for a stricter alternative.
    """
    preds = logits.argmax(dim=1)
    accs = []
    for c in range(n_classes):
        mask = (y == c)
        if mask.sum() > 0:
            accs.append((preds[mask] == c).float().mean().item())
    return float(np.mean(accs)) if accs else float("nan")


def _min_class_accuracy(logits, y, n_classes):
    """
    Minimum per-class recall across all classes, as a stricter
    alternative to macro-averaged recall. Directly closes the partial-
    collapse exploit documented in _macro_accuracy's docstring: a
    classifier that ignores even ONE class entirely scores 0.0 here,
    regardless of how well it does on the others. A configuration can
    only score well under this metric if it achieves non-trivial recall
    on every class simultaneously, which is a substantially stronger
    requirement than "the average across classes is above chance."
    """
    preds = logits.argmax(dim=1)
    accs = []
    for c in range(n_classes):
        mask = (y == c)
        if mask.sum() > 0:
            accs.append((preds[mask] == c).float().mean().item())
    return float(np.min(accs)) if accs else float("nan")


def evaluate_objectives(model, val_z, val_y, eval_cfg: EvalConfig, device="cpu",
                         metric: str = "macro"):
    """
    metric: "macro" (default) or "min"
            (stricter -- see _min_class_accuracy). Both are always
            computed and returned in the diagnostics dict so a search
            using one metric as its optimization objective can still
            report the other for comparison.
    """
    device = torch.device(device)
    model.eval()

    z = torch.as_tensor(val_z, dtype=torch.float32).to(device)
    y = torch.as_tensor(val_y, dtype=torch.long).to(device)
    n_classes = int(y.max().item()) + 1

    with torch.no_grad():
        logits_clean = model(z)
    g1_macro = _macro_accuracy(logits_clean, y, n_classes)
    g1_min = _min_class_accuracy(logits_clean, y, n_classes)

    with torch.no_grad():
        preds_clean = logits_clean.argmax(dim=1)
        nominal_mask = (y == NOMINAL_CLASS)
        if nominal_mask.sum() > 0:
            false_alarms = (preds_clean[nominal_mask] != NOMINAL_CLASS).float().mean().item()
        else:
            false_alarms = float("nan")
    g2_false_alarm_rate = false_alarms

    z_adv = pgd_attack(model, z, y, eps=eval_cfg.eps, alpha=eval_cfg.pgd_alpha,
                        steps=eval_cfg.pgd_steps_eval, norm=eval_cfg.norm)
    with torch.no_grad():
        logits_adv = model(z_adv)
    g3_macro = _macro_accuracy(logits_adv, y, n_classes)
    g3_min = _min_class_accuracy(logits_adv, y, n_classes)

    model.train()

    diagnostics = {"clean_macro": g1_macro, "clean_min": g1_min,
                    "robust_macro": g3_macro, "robust_min": g3_min}

    if metric == "min":
        return g1_min, g2_false_alarm_rate, g3_min, diagnostics
    return g1_macro, g2_false_alarm_rate, g3_macro, diagnostics
