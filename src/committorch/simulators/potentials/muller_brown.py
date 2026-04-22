r"""Muller-Brown potential energy surface.

The Muller-Brown potential is a standard 2D test surface with three minima
and two saddle points, widely used in rare-event method development.

.. math::
    V(x, y) = \sum_{k=0}^{3} A_k \exp\bigl[
        a_k (x - x_k^0)^2 + b_k (x - x_k^0)(y - y_k^0)
        + c_k (y - y_k^0)^2 \bigr]

Default parameters produce minima near (-0.558, 1.442), (0.623, 0.028),
and (-0.050, 0.467), with the deepest minimum at (-0.558, 1.442).

References
----------
.. [1] K. Muller and L. D. Brown, Theor. Chim. Acta 53, 75 (1979).
"""

import torch


_A = torch.tensor([-200.0, -100.0, -170.0, 15.0])
_a = torch.tensor([-1.0, -1.0, -6.5, 0.7])
_b = torch.tensor([0.0, 0.0, 11.0, 0.6])
_c = torch.tensor([-10.0, -10.0, -6.5, 0.7])
_x0 = torch.tensor([1.0, 0.0, -0.5, -1.0])
_y0 = torch.tensor([0.0, 0.5, 1.5, 1.0])


def energy(xy: torch.Tensor) -> torch.Tensor:
    """Compute Muller-Brown potential energy.

    Parameters
    ----------
    xy : torch.Tensor
        Positions with shape (..., 2).  Last dimension is (x, y).

    Returns
    -------
    torch.Tensor
        Potential energy with shape (...).
    """
    x = xy[..., 0]
    y = xy[..., 1]

    A = _A.to(xy.device, xy.dtype)
    a = _a.to(xy.device, xy.dtype)
    b = _b.to(xy.device, xy.dtype)
    c = _c.to(xy.device, xy.dtype)
    x0 = _x0.to(xy.device, xy.dtype)
    y0 = _y0.to(xy.device, xy.dtype)

    dx = x.unsqueeze(-1) - x0
    dy = y.unsqueeze(-1) - y0

    exponent = a * dx**2 + b * dx * dy + c * dy**2
    return (A * torch.exp(exponent)).sum(dim=-1)


def gradient(xy: torch.Tensor) -> torch.Tensor:
    """Compute analytical gradient of Muller-Brown potential.

    Parameters
    ----------
    xy : torch.Tensor
        Positions with shape (..., 2).

    Returns
    -------
    torch.Tensor
        Gradient dV/d(x,y) with same shape as input.
    """
    x = xy[..., 0]
    y = xy[..., 1]

    A = _A.to(xy.device, xy.dtype)
    a = _a.to(xy.device, xy.dtype)
    b = _b.to(xy.device, xy.dtype)
    c = _c.to(xy.device, xy.dtype)
    x0 = _x0.to(xy.device, xy.dtype)
    y0 = _y0.to(xy.device, xy.dtype)

    dx = x.unsqueeze(-1) - x0
    dy = y.unsqueeze(-1) - y0

    exponent = a * dx**2 + b * dx * dy + c * dy**2
    val = A * torch.exp(exponent)

    dV_dx = (val * (2.0 * a * dx + b * dy)).sum(dim=-1)
    dV_dy = (val * (b * dx + 2.0 * c * dy)).sum(dim=-1)

    return torch.stack([dV_dx, dV_dy], dim=-1)


REACTANT_MINIMUM = torch.tensor([-0.558, 1.442])
PRODUCT_MINIMUM = torch.tensor([0.623, 0.028])
INTERMEDIATE_MINIMUM = torch.tensor([-0.050, 0.467])
