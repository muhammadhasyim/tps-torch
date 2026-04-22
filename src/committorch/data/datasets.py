"""PyTorch datasets for committor training."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class CommittorDataset(Dataset):
    """Dataset of configurations with optional labels and weights.

    Parameters
    ----------
    configs : torch.Tensor, shape (N, dim) or (N, n_atoms, 3)
        Configuration data.
    labels : torch.Tensor or None, shape (N,)
        Committor labels (empirical estimates or boundary conditions).
    weights : torch.Tensor or None, shape (N,)
        Importance-sampling weights.
    """

    def __init__(
        self,
        configs: torch.Tensor,
        labels: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> None:
        self.configs = configs
        self.labels = labels
        self.weights = weights

    def __len__(self) -> int:
        return self.configs.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {"configs": self.configs[idx]}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        if self.weights is not None:
            item["weights"] = self.weights[idx]
        return item

    @classmethod
    def from_trajectory(
        cls,
        positions: torch.Tensor,
        labels: torch.Tensor | None = None,
        flatten: bool = True,
    ) -> CommittorDataset:
        """Create dataset from trajectory positions.

        Parameters
        ----------
        positions : torch.Tensor, shape (N, n_atoms, 3)
        labels : torch.Tensor or None
        flatten : bool
            If True, reshape (N, n_atoms, 3) -> (N, n_atoms*3).
        """
        if flatten and positions.dim() == 3:
            configs = positions.reshape(positions.shape[0], -1)
        else:
            configs = positions
        return cls(configs=configs, labels=labels)
