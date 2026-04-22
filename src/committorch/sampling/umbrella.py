r"""Committor-based umbrella sampling (Paper 1).

Each umbrella window alpha applies a harmonic bias on the committor value:

.. math::
    W_\alpha(x; \theta) = \frac{1}{2} \kappa_\alpha
        \bigl(\hat{q}(x;\theta) - q_\alpha\bigr)^2

where q_alpha are target committor values evenly spaced in [0, 1] and
kappa_alpha is the spring constant.

References
----------
.. [1] Hasyim, Batton, Mandadapu. arXiv:2107.13522, Section 3.1.
"""

from __future__ import annotations

import torch

from ..models.base import CommittorModel
from .base import BiasStrategy


class UmbrellaBias(BiasStrategy):
    """Harmonic umbrella bias on the committor value.

    Parameters
    ----------
    model : CommittorModel
        The committor network.
    target : float
        Target committor value q_alpha for this window.
    spring_constant : float
        Spring constant kappa_alpha.
    """

    def __init__(
        self,
        model: CommittorModel,
        target: float,
        spring_constant: float = 100.0,
    ) -> None:
        super().__init__(model)
        self.target = target
        self.spring_constant = spring_constant

    def bias_energy(self, x: torch.Tensor) -> torch.Tensor:
        q = self.model(x)
        return 0.5 * self.spring_constant * (q - self.target) ** 2


class UmbrellaWindows:
    """Collection of umbrella windows spanning [0, 1] in committor space.

    Parameters
    ----------
    model : CommittorModel
        The committor network.
    n_windows : int
        Number of windows.
    spring_constant : float
        Uniform spring constant for all windows.
    """

    def __init__(
        self,
        model: CommittorModel,
        n_windows: int,
        spring_constant: float = 100.0,
    ) -> None:
        self.targets = torch.linspace(0.0, 1.0, n_windows).tolist()
        self.windows = [
            UmbrellaBias(model, t, spring_constant) for t in self.targets
        ]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> UmbrellaBias:
        return self.windows[idx]
