"""Callbacks for the training loop: checkpointing, logging, rate estimation."""

from __future__ import annotations

from pathlib import Path

import torch

from ..models.base import CommittorModel
from ..analysis.rates import rate_from_bke_loss


def save_checkpoint(
    model: CommittorModel,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    loss: float,
    path: str | Path,
) -> None:
    """Save a training checkpoint.

    Parameters
    ----------
    model : CommittorModel
    optimizer : Optimizer
    iteration : int
    loss : float
    path : str or Path
    """
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: CommittorModel,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """Load a training checkpoint.

    Returns
    -------
    int
        The iteration number from the checkpoint.
    """
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["iteration"]


def estimate_rate_on_the_fly(
    bke_loss_value: float,
    kT: float,
    friction: float,
) -> float:
    """Estimate reaction rate from the current BKE loss value.

    Convenience wrapper around ``rate_from_bke_loss`` for use as a callback.
    """
    return rate_from_bke_loss(bke_loss_value, kT, friction)
