"""Communication backend abstraction for actor-learner architecture.

Provides a unified interface for sending/receiving model parameters
and data batches between learner and workers, whether using
torch.distributed, Ray, or single-process fallback.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class CommBackend(ABC):
    """Abstract communication backend.

    Subclasses implement the actual data transport between learner
    and worker processes.
    """

    @abstractmethod
    def broadcast_params(self, model: nn.Module) -> None:
        """Broadcast model parameters from learner to all workers.

        Parameters
        ----------
        model : nn.Module
            The model whose parameters are broadcast.
        """

    @abstractmethod
    def send_data(
        self,
        data: dict[str, torch.Tensor],
        dest: int = 0,
    ) -> None:
        """Send a data batch from a worker to the learner.

        Parameters
        ----------
        data : dict
            Keys like "configs", "weights", etc., mapping to tensors.
        dest : int
            Destination rank (typically 0 = learner).
        """

    @abstractmethod
    def recv_data(self, source: int | None = None) -> dict[str, torch.Tensor]:
        """Receive a data batch at the learner from any worker.

        Parameters
        ----------
        source : int or None
            Source rank.  None = any source.

        Returns
        -------
        dict of str -> Tensor
        """

    @abstractmethod
    def barrier(self) -> None:
        """Synchronization barrier."""

    @property
    @abstractmethod
    def rank(self) -> int:
        """Current process rank."""

    @property
    @abstractmethod
    def world_size(self) -> int:
        """Total number of processes."""

    @property
    def is_learner(self) -> bool:
        """Whether this process is the learner (rank 0)."""
        return self.rank == 0


class SingleProcessBackend(CommBackend):
    """Single-process fallback -- no actual communication.

    All operations are no-ops or pass-throughs.  Useful for debugging
    and testing without launching multiple processes.
    """

    def __init__(self) -> None:
        self._data_queue: list[dict[str, torch.Tensor]] = []

    def broadcast_params(self, model: nn.Module) -> None:
        pass

    def send_data(
        self,
        data: dict[str, torch.Tensor],
        dest: int = 0,
    ) -> None:
        self._data_queue.append({k: v.clone() for k, v in data.items()})

    def recv_data(self, source: int | None = None) -> dict[str, torch.Tensor]:
        if not self._data_queue:
            return {}
        return self._data_queue.pop(0)

    def barrier(self) -> None:
        pass

    @property
    def rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1


class TorchDistributedBackend(CommBackend):
    """Backend using torch.distributed (Gloo/NCCL).

    Parameters
    ----------
    backend_name : str
        "gloo" (CPU) or "nccl" (GPU).
    """

    def __init__(self, backend_name: str = "gloo") -> None:
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before creating "
                "TorchDistributedBackend. Call dist.init_process_group() first."
            )
        self._dist = dist
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()

    def broadcast_params(self, model: nn.Module) -> None:
        for param in model.parameters():
            self._dist.broadcast(param.data, src=0)

    def send_data(
        self,
        data: dict[str, torch.Tensor],
        dest: int = 0,
    ) -> None:
        keys = sorted(data.keys())
        for key in keys:
            self._dist.send(data[key].contiguous(), dst=dest)

    def recv_data(self, source: int | None = None) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "recv_data with dynamic shapes requires a handshake protocol. "
            "Use gather_data() instead for fixed-size batches."
        )

    def barrier(self) -> None:
        self._dist.barrier()

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size
