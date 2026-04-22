"""Abstract base class for enhanced sampling / biasing strategies.

A BiasStrategy computes a bias potential W(x) that modifies the sampling
distribution to improve coverage of transition regions.
"""

from abc import ABC, abstractmethod
from typing import Callable

import torch

from ..models.base import CommittorModel


class BiasStrategy(ABC):
    """Abstract bias strategy interface.

    Parameters
    ----------
    model : CommittorModel
        The committor network whose output defines the bias.
    """

    def __init__(self, model: CommittorModel) -> None:
        self.model = model

    @abstractmethod
    def bias_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute bias potential W(x).

        Parameters
        ----------
        x : torch.Tensor, shape (batch, dim)
            Configurations.

        Returns
        -------
        torch.Tensor, shape (batch,)
            Bias energy at each configuration.
        """

    def as_bias_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return a callable suitable for ``Simulator.set_bias``."""
        return self.bias_energy
