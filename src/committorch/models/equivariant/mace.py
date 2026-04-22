"""MACE backbone adapter for committor prediction.

Wraps a pre-trained MACE model, extracts the invariant node embeddings,
pools them to a graph-level feature vector, and feeds into the
committor head.

Requires: mace-torch (pip install mace-torch)

References
----------
.. [1] Batatia et al., MACE. NeurIPS 2022 / arXiv:2206.07697.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .backbone import PretrainedBackbone


class MACECommittor(PretrainedBackbone):
    """Committor model using a MACE encoder.

    This is a placeholder that documents the intended interface.
    The actual MACE loading requires mace-torch to be installed.

    Parameters
    ----------
    mace_model : nn.Module or None
        Pre-trained MACE model.  If None, a placeholder identity
        encoder is used (for testing).
    feature_dim : int
        Dimension of MACE's node-level invariant features after pooling.
    head_hidden_dims : tuple of int
        Committor head architecture.
    sigmoid_steepness : float
        Output sigmoid steepness.
    freeze_encoder : bool
        Whether to freeze the MACE encoder weights.
    """

    def __init__(
        self,
        mace_model: nn.Module | None = None,
        feature_dim: int = 128,
        head_hidden_dims: tuple[int, ...] = (32, 32),
        sigmoid_steepness: float = 1.0,
        freeze_encoder: bool = True,
    ) -> None:
        if mace_model is None:
            encoder = _PlaceholderEncoder(feature_dim)
        else:
            encoder = _MACEFeatureExtractor(mace_model, feature_dim)

        super().__init__(
            encoder=encoder,
            feature_dim=feature_dim,
            head_hidden_dims=head_hidden_dims,
            sigmoid_steepness=sigmoid_steepness,
            freeze_encoder=freeze_encoder,
        )


class _PlaceholderEncoder(nn.Module):
    """Placeholder for when mace-torch is not installed."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(1, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dummy = x.reshape(x.shape[0], -1).mean(dim=-1, keepdim=True)
        return self.linear(dummy)


class _MACEFeatureExtractor(nn.Module):
    """Extract invariant features from a MACE model.

    MACE produces per-atom features; this pools them to a
    graph-level (system-level) invariant vector.
    """

    def __init__(self, mace_model: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.mace = mace_model
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        node_features = self.mace(x)
        if node_features.dim() == 3:
            pooled = node_features.mean(dim=1)
        else:
            pooled = node_features
        return self.proj(pooled)
