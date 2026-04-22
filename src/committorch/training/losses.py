r"""Loss functions for committor training.

The core variational loss is the backward Kolmogorov equation (BKE) loss
from transition path theory (TPT):

.. math::
    L_v = \langle |\nabla_x q(x)|^2 \rangle

The exact minimizer is the true committor q*(x), and the minimum value
of L_v relates to the reaction rate:

.. math::
    \nu_R = \frac{k_B T}{\gamma} \langle |\nabla_x q|^2 \rangle

Boundary losses enforce q(x) = 0 for x in A and q(x) = 1 for x in B.

Supervised loss fits q to empirical estimates from short trajectory shooting.

References
----------
.. [1] Hasyim, Batton, Mandadapu. arXiv:2107.13522 (2022), Eqs. 4-7, 16.
.. [2] Trizio, Kang, Parrinello. arXiv:2410.17029 (2025), Methods.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..models.base import CommittorModel


class BKELoss(nn.Module):
    r"""Backward Kolmogorov equation variational loss.

    .. math::
        L_\mathrm{BKE} = \frac{1}{N} \sum_i w_i |\nabla_x q(x_i)|^2

    where w_i are importance-sampling weights (default 1).

    Parameters
    ----------
    model : CommittorModel
        The committor network.
    """

    def __init__(self, model: CommittorModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute BKE loss.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, dim)
            Configurations (need not require grad; handled internally).
        weights : torch.Tensor or None, shape (batch,)
            Importance-sampling weights.  If None, uniform weighting.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        grad_q = self.model.gradient(x)
        grad_sq = (grad_q**2).sum(dim=-1)
        if weights is not None:
            return (weights * grad_sq).mean()
        return grad_sq.mean()


class BoundaryLoss(nn.Module):
    r"""Boundary condition penalty loss.

    .. math::
        L_A = \frac{1}{N_A}\sum_{i \in A} q(x_i)^2, \quad
        L_B = \frac{1}{N_B}\sum_{i \in B} (q(x_i) - 1)^2

    Parameters
    ----------
    model : CommittorModel
        The committor network.
    """

    def __init__(self, model: CommittorModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute boundary loss.

        Parameters
        ----------
        x_a : torch.Tensor, shape (n_a, dim)
            Configurations in reactant basin A.
        x_b : torch.Tensor, shape (n_b, dim)
            Configurations in product basin B.

        Returns
        -------
        torch.Tensor
            Scalar loss = mean(q(x_a)^2) + mean((q(x_b) - 1)^2).
        """
        q_a = self.model(x_a)
        q_b = self.model(x_b)
        return (q_a**2).mean() + ((q_b - 1.0) ** 2).mean()


class SupervisedLoss(nn.Module):
    r"""Supervised loss from empirical committor estimates.

    Paper 1 defines two variants:
    - MSE: L = mean( (q(x) - q_emp(x))^2 )
    - Mean-error (preferred): averages the error per replica first,
      then squares, reducing noise from individual shot estimates.

    This implements the MSE variant.  Mean-error requires replica structure.

    Parameters
    ----------
    model : CommittorModel
        The committor network.
    """

    def __init__(self, model: CommittorModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        q_empirical: torch.Tensor,
    ) -> torch.Tensor:
        """Compute supervised MSE loss.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, dim)
            Configurations.
        q_empirical : torch.Tensor, shape (batch,)
            Empirical committor estimates from shooting trajectories.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        q_pred = self.model(x)
        return ((q_pred - q_empirical) ** 2).mean()


class CombinedLoss(nn.Module):
    """Weighted combination of BKE, boundary, and optional supervised losses.

    Parameters
    ----------
    model : CommittorModel
        The committor network.
    lambda_boundary : float
        Weight for boundary loss.
    lambda_supervised : float
        Weight for supervised loss (0 to disable).
    """

    def __init__(
        self,
        model: CommittorModel,
        lambda_boundary: float = 1.0,
        lambda_supervised: float = 0.0,
    ) -> None:
        super().__init__()
        self.bke = BKELoss(model)
        self.boundary = BoundaryLoss(model)
        self.supervised = SupervisedLoss(model) if lambda_supervised > 0 else None
        self.lambda_boundary = lambda_boundary
        self.lambda_supervised = lambda_supervised

    def forward(
        self,
        x: torch.Tensor,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        weights: torch.Tensor | None = None,
        q_empirical: torch.Tensor | None = None,
        x_supervised: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute combined loss.

        Parameters
        ----------
        x : torch.Tensor
            Bulk configurations for BKE loss.
        x_a, x_b : torch.Tensor
            Boundary configurations.
        weights : torch.Tensor or None
            Importance weights for BKE.
        q_empirical, x_supervised : torch.Tensor or None
            Supervised data (both must be provided if lambda_supervised > 0).

        Returns
        -------
        torch.Tensor
            Scalar total loss.
        """
        loss = self.bke(x, weights) + self.lambda_boundary * self.boundary(x_a, x_b)
        if (
            self.supervised is not None
            and q_empirical is not None
            and x_supervised is not None
        ):
            loss = loss + self.lambda_supervised * self.supervised(
                x_supervised, q_empirical
            )
        return loss
