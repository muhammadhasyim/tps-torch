"""Free energy surface (FES) computation from reweighted samples.

Constructs the FES along the committor q or latent z by histogramming
reweighted configurations.
"""

from __future__ import annotations

import torch


def fes_along_cv(
    cv_values: torch.Tensor,
    weights: torch.Tensor | None = None,
    n_bins: int = 100,
    kT: float = 1.0,
    cv_range: tuple[float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the free energy surface along a collective variable.

    .. math::
        F(s) = -k_B T \log P(s)

    where P(s) is the (reweighted) histogram of the CV.

    Parameters
    ----------
    cv_values : torch.Tensor, shape (N,)
        Collective variable values for each sample.
    weights : torch.Tensor or None, shape (N,)
        Importance-sampling weights.  If None, uniform.
    n_bins : int
        Number of histogram bins.
    kT : float
        Thermal energy.
    cv_range : tuple or None
        (min, max) range for the histogram.  If None, auto from data.

    Returns
    -------
    bin_centers : torch.Tensor, shape (n_bins,)
    free_energy : torch.Tensor, shape (n_bins,)
        Free energy relative to its minimum (minimum set to 0).
    """
    if cv_range is None:
        lo = cv_values.min().item()
        hi = cv_values.max().item()
    else:
        lo, hi = cv_range

    edges = torch.linspace(lo, hi, n_bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    idx = ((cv_values - lo) / (hi - lo + 1e-30) * n_bins).long().clamp(0, n_bins - 1)

    if weights is None:
        weights = torch.ones_like(cv_values)

    histogram = torch.zeros(n_bins, dtype=weights.dtype)
    histogram.scatter_add_(0, idx, weights)
    histogram = histogram / histogram.sum()

    histogram = histogram.clamp(min=1e-30)
    fes = -kT * torch.log(histogram)
    fes = fes - fes.min()

    return bin_centers, fes
