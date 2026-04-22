"""Transition state ensemble (TSE) analysis.

The TSE is the set of configurations with q(x) ~ 0.5.
"""

from __future__ import annotations

import torch

from ..models.base import CommittorModel


def extract_tse(
    model: CommittorModel,
    configs: torch.Tensor,
    q_window: tuple[float, float] = (0.45, 0.55),
) -> torch.Tensor:
    """Extract configurations in the transition state ensemble.

    Parameters
    ----------
    model : CommittorModel
    configs : torch.Tensor, shape (N, dim)
    q_window : tuple
        (q_min, q_max) defining the TSE region.

    Returns
    -------
    torch.Tensor
        Subset of configs with q in the TSE window.
    """
    with torch.no_grad():
        q = model(configs)
    mask = (q >= q_window[0]) & (q <= q_window[1])
    return configs[mask]
