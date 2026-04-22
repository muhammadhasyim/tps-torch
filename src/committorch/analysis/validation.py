"""Validation utilities for comparing committor predictions to references.

Provides tools to compare learned committors against analytical or
FEM reference solutions, primarily for the toy systems (1D quartic,
Muller-Brown).
"""

from __future__ import annotations

import torch

from ..models.base import CommittorModel


def mean_absolute_error(
    model: CommittorModel,
    x: torch.Tensor,
    q_reference: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute error between predicted and reference committor.

    Parameters
    ----------
    model : CommittorModel
        Trained model.
    x : torch.Tensor, shape (N, dim)
        Test configurations.
    q_reference : torch.Tensor, shape (N,)
        Reference committor values.

    Returns
    -------
    torch.Tensor
        Scalar MAE.
    """
    with torch.no_grad():
        q_pred = model(x)
    return (q_pred - q_reference).abs().mean()


def max_absolute_error(
    model: CommittorModel,
    x: torch.Tensor,
    q_reference: torch.Tensor,
) -> torch.Tensor:
    """Maximum absolute error."""
    with torch.no_grad():
        q_pred = model(x)
    return (q_pred - q_reference).abs().max()
