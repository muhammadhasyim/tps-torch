"""Abstract base class for simulators.

A Simulator drives sampling by stepping a system forward in time under
a potential V(x) plus an optional bias W(x).  It produces configurations
that are consumed by the training loop.
"""

from abc import ABC, abstractmethod
from typing import Callable

import torch


class Simulator(ABC):
    """Abstract simulator interface.

    Parameters
    ----------
    dim : int
        Dimensionality of the configuration space.
    kT : float
        Thermal energy k_B T in the same units as the potential.
    """

    def __init__(self, dim: int, kT: float = 1.0) -> None:
        self.dim = dim
        self.kT = kT
        self._bias_fn: Callable[[torch.Tensor], torch.Tensor] | None = None

    def set_bias(self, bias_fn: Callable[[torch.Tensor], torch.Tensor] | None) -> None:
        """Set (or clear) the bias potential W(x).

        Parameters
        ----------
        bias_fn : callable or None
            A function mapping configuration tensor (batch, dim) -> (batch,)
            returning the bias energy.  Pass None to remove the bias.
        """
        self._bias_fn = bias_fn

    @abstractmethod
    def step(self, x: torch.Tensor, n_steps: int = 1) -> torch.Tensor:
        """Advance configurations by *n_steps* integration steps.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, dim)
            Current configurations.
        n_steps : int
            Number of integration steps.

        Returns
        -------
        torch.Tensor, shape (batch, dim)
            Updated configurations.
        """

    @abstractmethod
    def potential_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute potential energy V(x).

        Parameters
        ----------
        x : torch.Tensor, shape (batch, dim) or (dim,)

        Returns
        -------
        torch.Tensor, shape (batch,) or scalar.
        """

    @abstractmethod
    def force(self, x: torch.Tensor) -> torch.Tensor:
        """Compute force -grad V(x).

        Parameters
        ----------
        x : torch.Tensor, shape (batch, dim) or (dim,)

        Returns
        -------
        torch.Tensor, same shape as *x*.
        """
