"""Adapter wrapping pre-trained MLIP encoders for committor prediction.

The backbone extracts invariant graph-level features from atomic
coordinates.  A committor head (small MLP -> sigmoid) is attached
on top.  The encoder can be frozen for transfer learning and later
fine-tuned with a lower learning rate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..base import CommittorModel


class PretrainedBackbone(CommittorModel):
    """Committor model built on a pre-trained MLIP backbone.

    Architecture::

        positions (N, 3) --> Encoder --> invariant features (F,)
                         --> CommittorHead (MLP) --> z --> sigmoid --> q

    Parameters
    ----------
    encoder : nn.Module
        Pre-trained encoder that maps atomic data to invariant features.
    feature_dim : int
        Dimension of the encoder's output feature vector.
    head_hidden_dims : Sequence[int]
        Hidden layer widths for the committor head MLP.
    sigmoid_steepness : float
        Steepness of the output sigmoid.
    freeze_encoder : bool
        If True, freeze encoder parameters (train only the head).
    """

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        head_hidden_dims: tuple[int, ...] = (32, 32),
        sigmoid_steepness: float = 1.0,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__(sigmoid_steepness=sigmoid_steepness)
        self.encoder = encoder
        self.feature_dim = feature_dim

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        layers: list[nn.Module] = []
        prev = feature_dim
        for h in head_hidden_dims:
            layers.append(nn.Linear(prev, h))
            # ReLU (not SiLU): openmm-torch force evaluation can raise
            # "derivative for aten::silu_backward is not implemented" on traced graphs.
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def _latent(self, x: torch.Tensor) -> torch.Tensor:
        """Compute z(x) = head(encoder(x)).

        The encoder handles both flat (batch, dim) and molecular
        (batch, n_atoms, 3) inputs transparently.
        """
        features = self.encoder(x)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        return self.head(features)

    def unfreeze_encoder(self, lr_scale: float = 0.1) -> list[dict]:
        """Unfreeze encoder for fine-tuning.

        Returns parameter groups suitable for an optimizer with
        differential learning rates.

        Parameters
        ----------
        lr_scale : float
            Scale factor for encoder learning rate relative to head.

        Returns
        -------
        list of dict
            Parameter groups: [{'params': encoder_params, 'lr_scale': lr_scale},
                               {'params': head_params}]
        """
        for param in self.encoder.parameters():
            param.requires_grad = True
        return [
            {"params": self.encoder.parameters(), "lr_scale": lr_scale},
            {"params": self.head.parameters()},
        ]


class DistanceEncoder(nn.Module):
    """Simple invariant encoder using pairwise distances (**test-only**).

    Computes all N*(N-1)/2 pairwise inter-atomic distances and passes them
    through a small MLP.  SE(3)-invariant by construction.

    .. warning::
        This encoder is intended **only** for unit tests, CI, and small
        toy systems (< 50 atoms).  The first Linear layer has N*(N-1)/2
        inputs, so memory scales as O(N^2).  For solvated proteins (thousands
        of atoms) this is prohibitively expensive and scientifically
        meaningless (water-water distances carry no folding information).

        For production use, see
        :class:`~committorch.models.equivariant.mace.MACECommittor` with a
        pre-trained MACE foundation model, which uses local atomic
        environments with a radial cutoff and scales as O(N).

    Parameters
    ----------
    n_atoms : int
        Number of atoms.
    hidden_dim : int
        Hidden dimension of the distance MLP.
    output_dim : int
        Dimension of the output feature vector.
    """

    def __init__(
        self,
        n_atoms: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n_atoms = n_atoms
        n_pairs = n_atoms * (n_atoms - 1) // 2
        self.mlp = nn.Sequential(
            nn.Linear(n_pairs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # register_buffer ensures _triu_indices is moved to the correct device
        # automatically when model.to(device) or model.cuda() is called, avoiding
        # CPU/CUDA device mismatches inside JIT-traced modules.
        self.register_buffer(
            "_triu_indices", torch.triu_indices(n_atoms, n_atoms, offset=1)
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute invariant features from positions.

        Parameters
        ----------
        positions : torch.Tensor, shape (batch, n_atoms, 3) or (batch, n_atoms*3)

        Returns
        -------
        torch.Tensor, shape (batch, output_dim)
        """
        # 2D layout is (batch, n_atoms * 3). Reshape using self.n_atoms so that
        # the shape is statically known (avoids any storage-access issues under
        # functorch transforms).
        if positions.dim() == 2:
            positions = positions.reshape(-1, self.n_atoms, 3)

        # torch.cdist uses the ||a||²+||b||²-2a·b identity so peak memory is
        # O(B*N²) instead of O(B*N²*3) for the explicit diff tensor.
        # upper-triangular distances (i≠j) are always positive so no sqrt-eps needed.
        dist_matrix = torch.cdist(positions, positions)
        # _triu_indices lives on the same device as model parameters (register_buffer).
        pairwise = dist_matrix[:, self._triu_indices[0], self._triu_indices[1]]
        return self.mlp(pairwise)
