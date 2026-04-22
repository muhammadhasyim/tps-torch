"""Abstract base class for committor function models.

A committor q(x) maps configurations x to [0, 1], giving the probability
of reaching product state B before reactant state A from configuration x.
Internally the model computes a latent z(x) and applies sigmoid: q = sigma(z).

References
----------
.. [1] Hasyim, Batton, Mandadapu. arXiv:2107.13522 (2022).
.. [2] Trizio, Kang, Parrinello. arXiv:2410.17029 (2025).
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class CommittorModel(ABC, nn.Module):
    """Abstract base for all committor neural networks.

    Subclasses must implement ``_latent`` which returns the pre-sigmoid
    scalar z(x).  The public API (``forward``, ``gradient``, ``latent``)
    is built on top and should rarely need overriding.

    Parameters
    ----------
    sigmoid_steepness : float
        The parameter *p* in sigma(z) = 1 / (1 + exp(-p*z)).
        Default 1.0 (standard sigmoid).  Paper 2 uses p=3.
    """

    def __init__(self, sigmoid_steepness: float = 1.0) -> None:
        super().__init__()
        self.sigmoid_steepness = sigmoid_steepness

    @abstractmethod
    def _latent(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the pre-sigmoid latent z(x).

        Parameters
        ----------
        x : torch.Tensor
            Configuration tensor.  Shape depends on the subclass:
            (batch, dim) for flat MLPs, (batch, n_atoms, 3) for
            equivariant models, etc.

        Returns
        -------
        torch.Tensor
            Shape (batch,) or (batch, 1) -- scalar per configuration.
        """

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        """Public interface to z(x), the pre-sigmoid output.

        Useful as a collective variable for OPES (Paper 2) since z has
        non-vanishing gradients everywhere unlike q which saturates in basins.
        """
        z = self._latent(x)
        return z.squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute committor q(x) = sigma(p * z(x)).

        Parameters
        ----------
        x : torch.Tensor
            Configuration tensor.

        Returns
        -------
        torch.Tensor
            Shape (batch,), values in (0, 1).
        """
        z = self.latent(x)
        return torch.sigmoid(self.sigmoid_steepness * z)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute grad_x q(x) via autograd.

        Parameters
        ----------
        x : torch.Tensor
            Configuration tensor with ``requires_grad=True``.

        Returns
        -------
        torch.Tensor
            Same shape as *x*, the spatial gradient of q w.r.t. x.
        """
        x = x.detach().requires_grad_(True)
        q = self.forward(x)
        (grad,) = torch.autograd.grad(
            q.sum(), x, create_graph=self.training
        )
        return grad

    def latent_gradient(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute grad_x z(x) via autograd.

        Needed for the numerically stable V_K formulation (Paper 2 SI).
        """
        x = x.detach().requires_grad_(True)
        z = self.latent(x)
        (grad,) = torch.autograd.grad(
            z.sum(), x, create_graph=self.training
        )
        return grad
