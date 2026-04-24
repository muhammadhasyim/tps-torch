"""PaiNN backbone adapter for committor prediction.

.. deprecated::
    This is a **placeholder stub** with no real PaiNN integration.
    Use :class:`~committorch.models.equivariant.mace.MACECommittor` with
    a pre-trained MACE foundation model instead.

References
----------
.. [1] Schutt et al., PaiNN. ICML 2021.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from .backbone import PretrainedBackbone


class PaiNNCommittor(PretrainedBackbone):
    """Committor model using a PaiNN encoder.

    .. deprecated::
        Placeholder adapter with no pre-trained model loading.
        Use ``MACECommittor.from_foundation()`` instead.
    """

    def __init__(
        self,
        painn_model: nn.Module | None = None,
        feature_dim: int = 64,
        head_hidden_dims: tuple[int, ...] = (32, 32),
        sigmoid_steepness: float = 1.0,
        freeze_encoder: bool = True,
    ) -> None:
        warnings.warn(
            "PaiNNCommittor is a deprecated placeholder with no real "
            "pre-trained model support. Use MACECommittor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
