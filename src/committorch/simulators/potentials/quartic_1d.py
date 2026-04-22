r"""Symmetric 1D double-well (quartic) potential.

.. math::
    V(x) = h \left(\frac{x^2}{a^2} - 1\right)^2

Two minima at x = +/- a with V = 0 and a barrier of height h at x = 0.
The exact committor for the overdamped Langevin dynamics with this potential
can be computed via quadrature:

.. math::
    q(x) = \frac{\int_{-a}^{x} e^{V(s)/k_BT} ds}
               {\int_{-a}^{a}  e^{V(s)/k_BT} ds}
"""

import torch


def energy(
    x: torch.Tensor,
    barrier_height: float = 2.0,
    well_position: float = 1.0,
) -> torch.Tensor:
    """Compute quartic double-well energy.

    Parameters
    ----------
    x : torch.Tensor
        1D position(s), any shape.
    barrier_height : float
        Height *h* of the barrier at x=0.
    well_position : float
        Position *a* of the two minima.

    Returns
    -------
    torch.Tensor
        Same shape as input.
    """
    return barrier_height * ((x / well_position) ** 2 - 1.0) ** 2


def gradient(
    x: torch.Tensor,
    barrier_height: float = 2.0,
    well_position: float = 1.0,
) -> torch.Tensor:
    """Analytical gradient dV/dx."""
    return (
        4.0
        * barrier_height
        * x
        * ((x / well_position) ** 2 - 1.0)
        / well_position**2
    )


def exact_committor(
    x: torch.Tensor,
    barrier_height: float = 2.0,
    well_position: float = 1.0,
    kT: float = 1.0,
    n_quad: int = 10000,
) -> torch.Tensor:
    """Compute the exact committor via numerical quadrature.

    Reactant A is at x = -well_position, product B at x = +well_position.

    Parameters
    ----------
    x : torch.Tensor
        Position(s) at which to evaluate the committor.
    barrier_height, well_position, kT : float
        Potential and thermal parameters.
    n_quad : int
        Number of quadrature points for the numerical integral.

    Returns
    -------
    torch.Tensor
        Same shape as *x*, values in [0, 1].
    """
    a = well_position
    s = torch.linspace(-a, a, n_quad, dtype=x.dtype, device=x.device)
    ds = s[1] - s[0]
    integrand = torch.exp(energy(s, barrier_height, well_position) / kT)
    cumulative = torch.cumsum(integrand, dim=0) * ds
    full_integral = cumulative[-1]

    x_flat = x.reshape(-1)
    idx = ((x_flat + a) / (2.0 * a) * (n_quad - 1)).clamp(0, n_quad - 1).long()
    q = cumulative[idx] / full_integral
    return q.reshape(x.shape)
