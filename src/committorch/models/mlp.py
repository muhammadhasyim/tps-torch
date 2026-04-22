"""Simple MLP committor model for low-dimensional systems.

Architecture matches the single-hidden-layer network from Hasyim et al.
(arXiv:2107.13522, Eq. in Section 5):

    z(x) = w2 . ReLU(W1 x + b1)
    q(x) = sigma(z(x))

Extended here to support multiple hidden layers and configurable activation.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .base import CommittorModel


class CommittorMLP(CommittorModel):
    """Multi-layer perceptron committor.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the configuration vector.
    hidden_dims : Sequence[int]
        Widths of hidden layers. Default (200,) matches the paper.
    activation : str
        Activation function name. Must be a valid ``torch.nn`` class name.
    sigmoid_steepness : float
        Steepness parameter *p* for the output sigmoid.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (200,),
        activation: str = "ReLU",
        sigmoid_steepness: float = 1.0,
    ) -> None:
        super().__init__(sigmoid_steepness=sigmoid_steepness)
        layers: list[nn.Module] = []
        act_cls = getattr(nn, activation)
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def _latent(self, x: torch.Tensor) -> torch.Tensor:
        """Compute z(x) = MLP(x) without sigmoid."""
        return self.net(x)
