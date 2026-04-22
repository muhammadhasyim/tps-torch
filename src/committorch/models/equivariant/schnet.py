"""SchNet backbone adapter for committor prediction.

SchNet uses continuous-filter convolutions on a graph of atoms,
producing invariant representations suitable for scalar prediction.

Requires: schnetpack (pip install schnetpack)

References
----------
.. [1] Schutt et al., SchNet. J. Chem. Phys. 148, 241722 (2018).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import PretrainedBackbone


class SchNetCommittor(PretrainedBackbone):
    """Committor model using a SchNet encoder.

    Placeholder adapter documenting the interface.  Actual SchNet loading
    requires schnetpack.

    Parameters
    ----------
    schnet_model : nn.Module or None
    feature_dim : int
    head_hidden_dims : tuple of int
    sigmoid_steepness : float
    freeze_encoder : bool
    """

    def __init__(
        self,
        schnet_model: nn.Module | None = None,
        feature_dim: int = 64,
        head_hidden_dims: tuple[int, ...] = (32, 32),
        sigmoid_steepness: float = 1.0,
        freeze_encoder: bool = True,
    ) -> None:
        if schnet_model is None:
            encoder = _PlaceholderEncoder(feature_dim)
        else:
            encoder = schnet_model

        super().__init__(
            encoder=encoder,
            feature_dim=feature_dim,
            head_hidden_dims=head_hidden_dims,
            sigmoid_steepness=sigmoid_steepness,
            freeze_encoder=freeze_encoder,
        )


class _PlaceholderEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(1, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dummy = x.reshape(x.shape[0], -1).mean(dim=-1, keepdim=True)
        return self.linear(dummy)
