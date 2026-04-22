r"""Overdamped Langevin dynamics integrator.

Integrates the overdamped Langevin equation:

.. math::
    dx = -\frac{1}{\gamma}\nabla V(x)\,dt + \sqrt{\frac{2 k_B T}{\gamma}} \, dW

using the Euler-Maruyama scheme:

.. math::
    x_{n+1} = x_n - \frac{\Delta t}{\gamma}\nabla [V + W](x_n)
              + \sqrt{\frac{2 k_B T \Delta t}{\gamma}} \, \xi_n

where xi_n ~ N(0, I).  Here W is an optional bias potential.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

from .base import Simulator


class OverdampedLangevin(Simulator):
    """Euler-Maruyama integrator for overdamped Langevin dynamics.

    Parameters
    ----------
    dim : int
        Configuration space dimension.
    potential_fn : callable
        Maps (batch, dim) -> (batch,) potential energy.
    gradient_fn : callable
        Maps (batch, dim) -> (batch, dim) gradient of V.
    kT : float
        Thermal energy.
    friction : float
        Friction coefficient gamma.
    dt : float
        Integration time step.
    """

    def __init__(
        self,
        dim: int,
        potential_fn: Callable[[torch.Tensor], torch.Tensor],
        gradient_fn: Callable[[torch.Tensor], torch.Tensor],
        kT: float = 1.0,
        friction: float = 1.0,
        dt: float = 1e-4,
    ) -> None:
        super().__init__(dim=dim, kT=kT)
        self._potential_fn = potential_fn
        self._gradient_fn = gradient_fn
        self.friction = friction
        self.dt = dt
        self._noise_scale = math.sqrt(2.0 * kT * dt / friction)

    def potential_energy(self, x: torch.Tensor) -> torch.Tensor:
        return self._potential_fn(x)

    def force(self, x: torch.Tensor) -> torch.Tensor:
        return -self._gradient_fn(x)

    def _total_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Gradient of V + W (if bias is set)."""
        grad = self._gradient_fn(x)
        if self._bias_fn is not None:
            x_req = x.detach().requires_grad_(True)
            w = self._bias_fn(x_req)
            (bias_grad,) = torch.autograd.grad(w.sum(), x_req)
            grad = grad + bias_grad.detach()
        return grad

    def step(self, x: torch.Tensor, n_steps: int = 1) -> torch.Tensor:
        """Advance configurations by Euler-Maruyama steps.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, dim)
            Current positions.
        n_steps : int
            Number of steps to take.

        Returns
        -------
        torch.Tensor, shape (batch, dim)
            Updated positions (detached from autograd graph).
        """
        for _ in range(n_steps):
            grad = self._total_gradient(x)
            with torch.no_grad():
                noise = torch.randn_like(x)
                x = x - (self.dt / self.friction) * grad + self._noise_scale * noise
                x = x.detach()
        return x
