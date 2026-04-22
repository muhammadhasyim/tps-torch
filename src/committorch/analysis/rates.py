r"""Reaction rate computation from the committor function.

From transition path theory, the reaction rate (frequency of A -> B
transitions) is:

.. math::
    \nu_R = \frac{k_B T}{\gamma} \langle |\nabla_x q(x)|^2 \rangle

where the average is over the Boltzmann distribution.

The BKE variational loss L_v = <|grad q|^2> is thus related to the
rate by:

.. math::
    \nu_R = \frac{k_B T}{\gamma} L_v

References
----------
.. [1] E, Vanden-Eijnden. J. Stat. Phys. 123, 503 (2006), Eq. for nu_R.
.. [2] Hasyim, Batton, Mandadapu. arXiv:2107.13522, Eq. 3.
"""

from __future__ import annotations

import torch

from ..models.base import CommittorModel


def rate_from_committor_gradient(
    model: CommittorModel,
    configs: torch.Tensor,
    kT: float,
    friction: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Estimate the reaction rate from sampled configurations.

    Parameters
    ----------
    model : CommittorModel
        Trained committor network.
    configs : torch.Tensor, shape (N, dim)
        Representative configurations sampled from the Boltzmann
        distribution (possibly reweighted).
    kT : float
        Thermal energy k_B T.
    friction : float
        Friction coefficient gamma.
    weights : torch.Tensor or None, shape (N,)
        Importance-sampling weights.  If None, assumes Boltzmann-distributed
        samples (uniform weights).

    Returns
    -------
    torch.Tensor
        Scalar estimated reaction rate.
    """
    grad_q = model.gradient(configs)
    grad_sq = (grad_q**2).sum(dim=-1)
    if weights is not None:
        mean_grad_sq = (weights * grad_sq).sum() / weights.sum()
    else:
        mean_grad_sq = grad_sq.mean()
    return (kT / friction) * mean_grad_sq


def rate_from_bke_loss(
    bke_loss_value: float,
    kT: float,
    friction: float,
) -> float:
    r"""Convert a BKE loss value to a reaction rate.

    Parameters
    ----------
    bke_loss_value : float
        Value of L_v = <|grad q|^2>.
    kT : float
        Thermal energy.
    friction : float
        Friction coefficient.

    Returns
    -------
    float
        Estimated reaction rate nu_R.
    """
    return (kT / friction) * bke_loss_value
