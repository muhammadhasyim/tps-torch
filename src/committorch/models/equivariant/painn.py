"""PaiNN backbone adapter for committor prediction.

PaiNN uses equivariant message passing with both scalar and vector
features, providing a good balance of expressiveness and efficiency.

References
----------
.. [1] Schutt et al., PaiNN. ICML 2021.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import PretrainedBackbone


class PaiNNCommittor(PretrainedBackbone):
    """Committor model using a PaiNN encoder.

    Placeholder adapter documenting the interface.

    Parameters
    ----------
    painn_model : nn.Module or None
    feature_dim : int
    head_hidden_dims : tuple of int
    sigmoid_steepness : float
    freeze_encoder : bool
    """

    def __init__(
        self,
        painn_model: nn.Module | None = None,
        feature_dim: int = 64,
        head_hidden_dims: tuple[int, ...] = (32, 32),
        sigmoid_steepness: float = 1.0,
        freeze_encoder: bool = True,
    ) -> None:
        if painn_model is None:
            encoder = _PlaceholderEncoder(feature_dim)
        else:
            encoder = painn_model

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
