"""Central learner process for actor-learner architecture.

The learner holds the model, accumulates gradient batches from workers,
updates parameters, and broadcasts new weights.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim

from ..models.base import CommittorModel
from ..training.losses import BKELoss, BoundaryLoss
from .comm import CommBackend


class Learner:
    """Central model holder and trainer.

    Parameters
    ----------
    model : CommittorModel
        The committor network (lives on one device).
    comm : CommBackend
        Communication backend.
    lr : float
        Learning rate.
    lambda_boundary : float
        Weight for boundary loss.
    x_a : torch.Tensor
        Reactant basin samples.
    x_b : torch.Tensor
        Product basin samples.
    device : str or torch.device
        Device for the model.
    """

    def __init__(
        self,
        model: CommittorModel,
        comm: CommBackend,
        lr: float = 1e-3,
        lambda_boundary: float = 1.0,
        x_a: torch.Tensor | None = None,
        x_b: torch.Tensor | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.comm = comm
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.bke_loss_fn = BKELoss(model)
        self.boundary_loss_fn = BoundaryLoss(model)
        self.lambda_boundary = lambda_boundary

        self.x_a = x_a.to(device) if x_a is not None else None
        self.x_b = x_b.to(device) if x_b is not None else None

        self._loss_history: list[float] = []

    def broadcast_params(self) -> None:
        """Send current model parameters to all workers."""
        self.comm.broadcast_params(self.model)

    def train_on_batch(
        self,
        configs: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> float:
        """Perform one gradient update on a data batch.

        Parameters
        ----------
        configs : torch.Tensor, shape (batch, dim)
        weights : torch.Tensor or None, shape (batch,)

        Returns
        -------
        float
            Total loss value.
        """
        configs = configs.to(self.device)
        if weights is not None:
            weights = weights.to(self.device)

        self.optimizer.zero_grad()
        bke = self.bke_loss_fn(configs, weights)

        loss = bke
        if self.x_a is not None and self.x_b is not None:
            bc = self.boundary_loss_fn(self.x_a, self.x_b)
            loss = loss + self.lambda_boundary * bc

        loss.backward()
        self.optimizer.step()

        loss_val = loss.item()
        self._loss_history.append(loss_val)
        return loss_val

    def collect_and_train(self, n_workers: int = 1) -> float:
        """Collect data from workers and train.

        For single-process mode, expects data to be in the comm queue.

        Parameters
        ----------
        n_workers : int
            Number of worker data batches to collect.

        Returns
        -------
        float
            Loss value.
        """
        all_configs = []
        all_weights = []

        for _ in range(n_workers):
            data = self.comm.recv_data()
            if not data:
                continue
            all_configs.append(data.get("configs", torch.empty(0)))
            if "weights" in data:
                all_weights.append(data["weights"])

        if not all_configs or all(c.numel() == 0 for c in all_configs):
            return 0.0

        configs = torch.cat(all_configs, dim=0)
        weights = torch.cat(all_weights, dim=0) if all_weights else None
        return self.train_on_batch(configs, weights)

    @property
    def loss_history(self) -> list[float]:
        return list(self._loss_history)
