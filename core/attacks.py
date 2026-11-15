"""
attacks.py

Projected gradient descent (PGD) adversarial perturbation, applied in
ILR coordinates so that perturbed points map back to valid compositions
under ilr_inv.
"""

import torch
import torch.nn.functional as F


def pgd_attack(model, z: torch.Tensor, y: torch.Tensor, eps: float,
                alpha: float, steps: int, norm: str = "linf") -> torch.Tensor:
    """
    Find z' = z + delta, ||delta||_p <= eps, that (approximately)
    maximizes the classification loss for `model`.

    z: (N, D-1) clean ILR inputs (will not be modified in place)
    y: (N,) true labels
    eps: perturbation budget in ILR space
    alpha: PGD step size
    steps: number of PGD iterations
    norm: "linf" or "l2"
    """
    model.eval()
    z = z.detach()
    delta = torch.zeros_like(z).uniform_(-eps, eps)
    if norm == "l2":
        delta = _project_l2(delta, eps)
    delta.requires_grad_(True)

    for _ in range(steps):
        logits = model(z + delta)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, delta)[0]

        with torch.no_grad():
            if norm == "linf":
                delta += alpha * grad.sign()
                delta.clamp_(-eps, eps)
            elif norm == "l2":
                grad_norm = grad.view(grad.size(0), -1).norm(dim=1).clamp_min(1e-12)
                grad_dir = grad / grad_norm.view(-1, 1)
                delta += alpha * grad_dir
                delta = _project_l2(delta, eps)
            else:
                raise ValueError(f"unknown norm {norm}")
        delta = delta.detach().requires_grad_(True)

    model.train()
    return (z + delta).detach()


def _project_l2(delta: torch.Tensor, eps: float) -> torch.Tensor:
    norm = delta.view(delta.size(0), -1).norm(dim=1).clamp_min(1e-12)
    factor = torch.clamp(eps / norm, max=1.0)
    return delta * factor.view(-1, 1)
