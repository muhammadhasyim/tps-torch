"""Export committor models for use with OpenMM's TorchForce.

``CommittorBiasForOpenMM`` wraps a committor model into a module whose
``forward(positions) -> scalar`` interface matches what TorchForce expects.
It can be traced with ``torch.jit.trace`` and saved as a ``.pt`` file.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import CommittorModel


def _kolmogorov_bias_energy(
    model: CommittorModel,
    pos: torch.Tensor,
    lam: float,
    beta: float,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Kolmogorov V_K via ``torch.autograd.grad`` + ``enable_grad`` (JIT-trace-safe).

    Replaces the previous ``jacrev``-based version: ``jacrev`` uses ``vmap``
    internally which hard-codes CPU device for its Jacobian accumulation buffers,
    causing a device mismatch when the model is on CUDA.  ``autograd.grad`` uses a
    single VJP (one backward pass) with no ``vmap``, is fully JIT-traceable, and
    works on any device.  ``create_graph=True`` keeps ``grad_z`` in the computation
    graph so TorchForce can differentiate through the energy to compute forces.
    """
    import torch.nn.functional as F

    param_dtype = next(model.parameters()).dtype
    param_device = next(model.parameters()).device
    x = pos.reshape(1, -1).to(device=param_device, dtype=param_dtype)

    with torch.enable_grad():
        x_grad = x.detach().requires_grad_(True)
        z = model.latent(x_grad)
        (grad_z,) = torch.autograd.grad(z.sum(), x_grad, create_graph=True)

    grad_z_sq = (grad_z**2).sum()
    p = model.sigmoid_steepness
    log_grad_z_sq = torch.log(grad_z_sq + eps)
    softplus_term = 4.0 * F.softplus(-p * z)
    linear_term = 2.0 * p * z
    return (lam / beta) * (log_grad_z_sq - softplus_term - linear_term).squeeze()


class CommittorBiasForOpenMM(nn.Module):
    """Export wrapper for using a committor model as a TorchForce bias.

    Takes raw flattened positions from OpenMM and computes a scalar
    bias energy.  Handles internal reshaping and graph construction
    (if using a GNN-based committor).

    Parameters
    ----------
    model : CommittorModel
        The committor model to wrap.
    bias_type : str
        Type of bias: "umbrella", "kolmogorov", "none".
    bias_params : dict or None
        Bias-specific parameters.
    n_atoms : int
        Number of atoms in the system.
    """

    def __init__(
        self,
        model: CommittorModel,
        bias_type: str = "umbrella",
        bias_params: dict | None = None,
        n_atoms: int = 0,
    ) -> None:
        super().__init__()
        self.model = model
        self.bias_type = bias_type
        self.bias_params = bias_params or {}
        self.n_atoms = n_atoms

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute bias energy from flat positions.

        Parameters
        ----------
        positions : torch.Tensor
            Shape (n_atoms, 3) or (n_atoms*3,) from OpenMM.

        Returns
        -------
        torch.Tensor
            Scalar bias energy in kJ/mol.
        """
        param_dtype = next(self.model.parameters()).dtype
        param_device = next(self.model.parameters()).device
        # Cast both dtype and device: OpenMM CUDA passes float64 CUDA tensors.
        positions = positions.to(device=param_device, dtype=param_dtype)
        pos = positions.reshape(1, -1)

        if self.bias_type == "umbrella":
            q = self.model(pos)
            target = self.bias_params.get("target", 0.5)
            kappa = self.bias_params.get("spring_constant", 100.0)
            return 0.5 * kappa * (q.squeeze() - target) ** 2
        elif self.bias_type == "kolmogorov":
            lam = self.bias_params.get("lambda", 1.0)
            beta = self.bias_params.get("beta", 1.0)
            eps = self.bias_params.get("eps", 1e-10)
            return _kolmogorov_bias_energy(
                self.model, pos, lam=lam, beta=beta, eps=eps
            )
        else:
            return torch.tensor(0.0, dtype=param_dtype, device=param_device)

    def export(self, path: str | Path) -> Path:
        """Trace and save the module as a TorchScript file.

        Parameters
        ----------
        path : str or Path
            Output file path (should end in .pt).

        Returns
        -------
        Path
            The saved file path.
        """
        path = Path(path)
        self.eval()
        param_device = next(self.model.parameters()).device
        example_input = torch.randn(self.n_atoms, 3, device=param_device)
        with torch.no_grad():
            traced = torch.jit.trace(self, example_input)
        traced.save(str(path))
        return path
