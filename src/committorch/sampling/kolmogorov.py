r"""Kolmogorov bias V_K (Paper 2).

The Kolmogorov bias concentrates sampling near the transition state by
adding a potential that depends on the committor gradient:

.. math::
    V_K(x) = -\frac{\lambda}{\beta} \log |\nabla_x q(x)|^2

In terms of the pre-sigmoid latent z(x) (numerically stable form from
Paper 2 SI):

.. math::
    V_K(x) = \frac{\lambda}{\beta}\bigl[
        \log|\nabla_x z(x)|^2
        - 4\log(1 + e^{-p z(x)})
        - 2 p z(x)
    \bigr] + \text{const}

where p is the sigmoid steepness.

References
----------
.. [1] Trizio, Kang, Parrinello. arXiv:2410.17029, Eq. in SI.
.. [2] Kang et al., Nat. Comput. Sci. 4, 451 (2024).
"""

from __future__ import annotations

import torch

from ..models.base import CommittorModel
from .base import BiasStrategy


class KolmogorovBias(BiasStrategy):
    r"""Kolmogorov bias V_K using the numerically stable z-based formulation.

    Parameters
    ----------
    model : CommittorModel
        Committor network.
    lam : float
        Bias strength lambda (typically 0.3--1.0).
    beta : float
        Inverse temperature 1/(k_B T).
    eps : float
        Small constant to prevent log(0).
    """

    def __init__(
        self,
        model: CommittorModel,
        lam: float = 1.0,
        beta: float = 1.0,
        eps: float = 1e-10,
    ) -> None:
        super().__init__(model)
        self.lam = lam
        self.beta = beta
        self.eps = eps

    def bias_energy(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute V_K(x) using the stable z-based formula.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, dim)
            Must have requires_grad=True if autograd-based force is needed.

        Returns
        -------
        torch.Tensor, shape (batch,)
        """
        needs_own_grad = not x.requires_grad
        if needs_own_grad:
            x = x.detach().requires_grad_(True)

        z = self.model.latent(x)
        (grad_z,) = torch.autograd.grad(
            z.sum(), x, create_graph=True, retain_graph=True
        )
        grad_z_sq = (grad_z**2).sum(dim=-1)

        p = self.model.sigmoid_steepness
        log_grad_z_sq = torch.log(grad_z_sq + self.eps)
        softplus_term = 4.0 * torch.nn.functional.softplus(-p * z)
        linear_term = 2.0 * p * z

        vk = (self.lam / self.beta) * (log_grad_z_sq - softplus_term - linear_term)
        return vk

    def transition_state_density(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute the Kolmogorov probability density p_K(x).

        p_K is proportional to rho(x) * |grad q|^2, which concentrates
        on the transition state ensemble.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, dim)

        Returns
        -------
        torch.Tensor, shape (batch,)
            Unnormalized density.
        """
        grad_q = self.model.gradient(x)
        return (grad_q**2).sum(dim=-1)
