"""Validation utilities for comparing committor predictions to references.

Provides tools to compare learned committors against:
- Analytical or FEM reference solutions (toy systems)
- DESRES brute-force trajectories (protein folding)

Includes state assignment accuracy, rate comparison, and
committor-along-trajectory evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from ..models.base import CommittorModel

logger = logging.getLogger(__name__)


def mean_absolute_error(
    model: CommittorModel,
    x: torch.Tensor,
    q_reference: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute error between predicted and reference committor.

    Parameters
    ----------
    model : CommittorModel
        Trained model.
    x : torch.Tensor, shape (N, dim)
        Test configurations.
    q_reference : torch.Tensor, shape (N,)
        Reference committor values.

    Returns
    -------
    torch.Tensor
        Scalar MAE.
    """
    with torch.no_grad():
        q_pred = model(x)
    return (q_pred - q_reference).abs().mean()


def max_absolute_error(
    model: CommittorModel,
    x: torch.Tensor,
    q_reference: torch.Tensor,
) -> torch.Tensor:
    """Maximum absolute error."""
    with torch.no_grad():
        q_pred = model(x)
    return (q_pred - q_reference).abs().max()


def validate_committor_vs_brute_force(
    model: CommittorModel,
    rmsd_trace: torch.Tensor,
    committor_values: torch.Tensor,
    folded_cutoff: float = 0.2,
) -> dict[str, float]:
    """Compare committor state assignments with RMSD-based labels.

    Parameters
    ----------
    model : CommittorModel
        Trained committor (not used directly; committor_values pre-computed).
    rmsd_trace : torch.Tensor, shape (n_frames,)
        RMSD to native structure.
    committor_values : torch.Tensor, shape (n_frames,)
        q(x) for each frame.
    folded_cutoff : float
        RMSD threshold for "folded" in nm.

    Returns
    -------
    dict with:
        "accuracy": fraction of frames where q-based and RMSD-based labels agree
        "mean_q_folded": mean committor for RMSD-folded frames
        "mean_q_unfolded": mean committor for RMSD-unfolded frames
        "n_frames": total number of frames
    """
    folded_mask = rmsd_trace < folded_cutoff
    committor_folded = committor_values < 0.5

    accuracy = (folded_mask == committor_folded).float().mean().item()

    mean_q_folded = (
        committor_values[folded_mask].mean().item()
        if folded_mask.any() else float("nan")
    )
    mean_q_unfolded = (
        committor_values[~folded_mask].mean().item()
        if (~folded_mask).any() else float("nan")
    )

    return {
        "accuracy": accuracy,
        "mean_q_folded": mean_q_folded,
        "mean_q_unfolded": mean_q_unfolded,
        "n_frames": len(rmsd_trace),
    }


def committor_rate_vs_brute_force(
    committor_values: torch.Tensor,
    rmsd_trace: torch.Tensor,
    time: torch.Tensor,
    folded_cutoff: float = 0.2,
    unfolded_cutoff: float = 0.6,
    committor_threshold: float = 0.5,
) -> dict[str, float]:
    """Compare folding rates from committor-based and RMSD-based assignments.

    Parameters
    ----------
    committor_values : torch.Tensor, shape (n_frames,)
        q(x) for each frame.
    rmsd_trace : torch.Tensor, shape (n_frames,)
        RMSD to native for each frame.
    time : torch.Tensor, shape (n_frames,)
        Time for each frame.
    folded_cutoff : float
        RMSD threshold for folded.
    unfolded_cutoff : float
        RMSD threshold for unfolded.
    committor_threshold : float
        q(x) threshold for "folded" state.

    Returns
    -------
    dict with rates from both methods and their ratio.
    """
    from ..data.desres import compute_brute_force_rate

    bf = compute_brute_force_rate(
        rmsd_trace, time, folded_cutoff, unfolded_cutoff
    )

    is_folded_q = committor_values < committor_threshold
    folding_events_q = 0
    state = "unknown"
    transition_times_q: list[float] = []
    last_t = time[0].item()

    for i in range(len(committor_values)):
        if is_folded_q[i] and state != "folded":
            if state == "unfolded":
                transition_times_q.append(time[i].item() - last_t)
                folding_events_q += 1
            state = "folded"
            last_t = time[i].item()
        elif not is_folded_q[i] and state != "unfolded":
            if state == "folded":
                pass
            state = "unfolded"
            last_t = time[i].item()

    total_time = (time[-1] - time[0]).item()
    committor_rate = folding_events_q / total_time if total_time > 0 else 0.0

    ratio = (
        committor_rate / bf["folding_rate"]
        if bf["folding_rate"] > 0 else float("inf")
    )

    return {
        "brute_force_rate": bf["folding_rate"],
        "brute_force_mfpt": bf["mean_folding_time"],
        "brute_force_n_events": bf["n_folding_events"],
        "committor_rate": committor_rate,
        "committor_n_events": folding_events_q,
        "rate_ratio": ratio,
        "total_time": total_time,
    }


def plot_committor_along_trajectory(
    committor_values: torch.Tensor,
    rmsd_trace: torch.Tensor,
    time: torch.Tensor | None = None,
    folded_cutoff: float = 0.2,
    output_path: str | Path | None = None,
) -> None:
    """Plot q(t) overlaid with RMSD-based state labels.

    Parameters
    ----------
    committor_values : torch.Tensor, shape (n_frames,)
    rmsd_trace : torch.Tensor, shape (n_frames,)
    time : torch.Tensor or None, shape (n_frames,)
        If None, uses frame indices.
    folded_cutoff : float
        RMSD threshold for folded state shading.
    output_path : str, Path, or None
        If provided, saves the figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plot")
        return

    if time is None:
        time = torch.arange(len(committor_values), dtype=torch.float64)

    t = time.numpy()
    q = committor_values.numpy()
    r = rmsd_trace.numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(t, q, linewidth=0.5, color="blue", alpha=0.7)
    ax1.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="q = 0.5")
    ax1.set_ylabel("Committor q(x)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()

    ax2.plot(t, r, linewidth=0.5, color="green", alpha=0.7)
    ax2.axhline(folded_cutoff, color="red", linestyle="--", alpha=0.5,
                label=f"folded cutoff = {folded_cutoff} nm")
    ax2.set_ylabel("RMSD to native (nm)")
    ax2.set_xlabel("Time")
    ax2.legend()

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("Saved plot to %s", output_path)
    else:
        plt.show()

    plt.close(fig)
