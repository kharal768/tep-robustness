"""
coda.py

Compositional data (CoDa) utilities: isometric log-ratio (ILR) transform
and its inverse, using an orthonormal Helmert-type basis.

All classifier training, adversarial perturbation, and optimization in
this pipeline operate in ILR (real, unconstrained) coordinates.
Raw compositions are recovered with
ilr_inv only where a compositional quantity is actually needed.
"""

import numpy as np
import torch


def _helmert_ilr_basis(D: int) -> np.ndarray:
    """
    Build a (D-1) x D orthonormal basis (in the Aitchison/clr sense) for
    the ILR transform, following the standard Helmert-type construction.

    Returns V such that ilr(x) = V @ clr(x), with V V^T = I_{D-1}.
    """
    V = np.zeros((D - 1, D))
    for i in range(D - 1):
        k = i + 1
        val = np.sqrt(1.0 / (k * (k + 1)))
        V[i, :k] = val
        V[i, k] = -k * val
    return V


class ILRTransform:
    """
    Isometric log-ratio transform for D-part compositions.

    x: (..., D) array of strictly positive compositions (need not be
       pre-closed to a fixed sum; closure is applied internally).
    z: (..., D-1) array of ILR coordinates in real space.
    """

    def __init__(self, D: int):
        self.D = D
        self.V = _helmert_ilr_basis(D)  # (D-1, D)
        self.V_pinv = np.linalg.pinv(self.V)  # (D, D-1)

    def close(self, x: np.ndarray, kappa: float = 1.0) -> np.ndarray:
        """Closure operation: rescale parts to sum to kappa."""
        x = np.clip(x, 1e-12, None)
        return kappa * x / x.sum(axis=-1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x (..., D) -> z (..., D-1)"""
        x = np.clip(x, 1e-12, None)
        logx = np.log(x)
        clr = logx - logx.mean(axis=-1, keepdims=True)
        z = clr @ self.V.T
        return z

    def inverse(self, z: np.ndarray, kappa: float = 1.0) -> np.ndarray:
        """z (..., D-1) -> x (..., D), closed to sum kappa."""
        clr = z @ self.V_pinv.T
        x = np.exp(clr)
        return self.close(x, kappa=kappa)

    # ---- torch-differentiable versions, needed for PGD in ILR space ----

    def forward_t(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=1e-12)
        logx = torch.log(x)
        clr = logx - logx.mean(dim=-1, keepdim=True)
        V = torch.as_tensor(self.V, dtype=x.dtype, device=x.device)
        return clr @ V.T

    def inverse_t(self, z: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
        V_pinv = torch.as_tensor(self.V_pinv, dtype=z.dtype, device=z.device)
        clr = z @ V_pinv.T
        x = torch.exp(clr)
        x = torch.clamp(x, min=1e-12)
        return kappa * x / x.sum(dim=-1, keepdim=True)


if __name__ == "__main__":
    # Sanity check: forward/inverse should round-trip.
    rng = np.random.default_rng(0)
    D = 6
    t = ILRTransform(D)
    x = rng.dirichlet(alpha=np.ones(D), size=5)
    z = t.forward(x)
    x_rec = t.inverse(z)
    err = np.abs(x - x_rec).max()
    print(f"D={D}, max reconstruction error = {err:.2e}")
    assert err < 1e-8, "ILR round-trip failed"
    print("ILR forward/inverse round-trip OK.")
