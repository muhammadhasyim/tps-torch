"""DESRES fast-folder trajectory loader and brute-force rate computation.

Handles loading and analysis of the millisecond-timescale MD trajectories
from Lindorff-Larsen et al., Science 334, 517 (2011).

The DESRES fast-folding protein dataset contains trajectories for 12 small
proteins simulated on the Anton supercomputer.  We focus on the four with
the most folding/unfolding events for benchmarking:

1. **CLN025** (chignolin variant): ~0.6 us folding time, 106 us trajectory
2. **Trp-cage**: ~4 us folding time, 208 us trajectory
3. **BBA**: ~18 us folding time, 325 us trajectory
4. **Villin**: ~0.7 us folding time, 125 us trajectory

Data availability
-----------------
DESRES trajectories must be obtained from D. E. Shaw Research:
- Check https://www.deshawresearch.com/downloads/
- Or request via the BioEmu benchmark repository:
  https://github.com/microsoft/bioemu-benchmarks

References
----------
.. [1] Lindorff-Larsen et al. Science 334, 517 (2011).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .trajectory import Trajectory, load_trajectory
from .features import rmsd


@dataclass
class FoldingProtein:
    """Metadata for a DESRES fast-folding protein.

    Parameters
    ----------
    name : str
        Protein identifier.
    pdb_id : str
        PDB accession code for the native structure.
    n_residues : int
        Number of residues.
    experimental_folding_time_us : float
        Experimental folding time in microseconds.
    trajectory_length_us : float
        Total simulation length in microseconds.
    rmsd_folded_cutoff_nm : float
        RMSD cutoff below which the protein is considered folded.
    rmsd_unfolded_cutoff_nm : float
        RMSD cutoff above which the protein is considered unfolded.
    """

    name: str
    pdb_id: str
    n_residues: int
    experimental_folding_time_us: float
    trajectory_length_us: float
    rmsd_folded_cutoff_nm: float = 0.2
    rmsd_unfolded_cutoff_nm: float = 0.6


PROTEINS = {
    "CLN025": FoldingProtein(
        name="CLN025",
        pdb_id="5AWL",
        n_residues=10,
        experimental_folding_time_us=0.6,
        trajectory_length_us=106.0,
    ),
    "Trp-cage": FoldingProtein(
        name="Trp-cage",
        pdb_id="2JOF",
        n_residues=20,
        experimental_folding_time_us=4.1,
        trajectory_length_us=208.0,
    ),
    "BBA": FoldingProtein(
        name="BBA",
        pdb_id="1FME",
        n_residues=28,
        experimental_folding_time_us=18.0,
        trajectory_length_us=325.0,
    ),
    "villin": FoldingProtein(
        name="villin",
        pdb_id="2F4K",
        n_residues=35,
        experimental_folding_time_us=0.7,
        trajectory_length_us=125.0,
    ),
}


def load_desres_trajectory(
    trajectory_dir: str | Path,
    protein: str | FoldingProtein,
    topology_file: str | Path | None = None,
    stride: int = 100,
    ca_only: bool = True,
) -> Trajectory:
    """Load a DESRES fast-folding protein trajectory.

    Parameters
    ----------
    trajectory_dir : str or Path
        Directory containing trajectory files (.dcd or .xtc).
    protein : str or FoldingProtein
        Protein name (must be in PROTEINS) or FoldingProtein instance.
    topology_file : str or Path or None
        Path to topology file (.pdb).
    stride : int
        Frame stride for subsampling.
    ca_only : bool
        If True, load only C-alpha atoms.

    Returns
    -------
    Trajectory
    """
    if isinstance(protein, str):
        if protein not in PROTEINS:
            raise ValueError(
                f"Unknown protein '{protein}'. Available: {list(PROTEINS.keys())}"
            )
        protein = PROTEINS[protein]

    traj_dir = Path(trajectory_dir)
    traj_files = sorted(traj_dir.glob("*.dcd")) + sorted(traj_dir.glob("*.xtc"))

    if not traj_files:
        raise FileNotFoundError(
            f"No trajectory files found in {traj_dir}. "
            f"DESRES data must be downloaded separately."
        )

    if topology_file is None:
        pdb_files = sorted(traj_dir.glob("*.pdb"))
        if pdb_files:
            topology_file = pdb_files[0]
        else:
            raise FileNotFoundError(
                f"No topology (.pdb) file found in {traj_dir}. "
                f"Provide topology_file explicitly."
            )

    traj = load_trajectory(
        str(traj_files[0]),
        topology_path=str(topology_file),
        stride=stride,
    )
    traj.metadata["protein"] = protein.name
    return traj


def compute_brute_force_rate(
    rmsd_trace: torch.Tensor,
    time: torch.Tensor,
    folded_cutoff: float = 0.2,
    unfolded_cutoff: float = 0.6,
) -> dict[str, float]:
    """Compute folding rate from direct counting of transitions.

    Identifies folding (unfolded->folded) and unfolding (folded->unfolded)
    events by tracking RMSD crossings.

    Parameters
    ----------
    rmsd_trace : torch.Tensor, shape (n_frames,)
        RMSD to native structure over time.
    time : torch.Tensor, shape (n_frames,)
        Simulation time for each frame (in same units throughout).
    folded_cutoff : float
        RMSD below this = folded.
    unfolded_cutoff : float
        RMSD above this = unfolded.

    Returns
    -------
    dict with keys:
        "n_folding_events": int
        "n_unfolding_events": int
        "mean_folding_time": float (in time units)
        "mean_unfolding_time": float
        "folding_rate": float (1/time)
        "unfolding_rate": float (1/time)
    """
    is_folded = rmsd_trace < folded_cutoff
    is_unfolded = rmsd_trace > unfolded_cutoff

    folding_times: list[float] = []
    unfolding_times: list[float] = []

    state = "unknown"
    last_transition_time = time[0].item()

    for i in range(len(rmsd_trace)):
        if is_folded[i] and state != "folded":
            if state == "unfolded":
                folding_times.append(time[i].item() - last_transition_time)
            state = "folded"
            last_transition_time = time[i].item()
        elif is_unfolded[i] and state != "unfolded":
            if state == "folded":
                unfolding_times.append(time[i].item() - last_transition_time)
            state = "unfolded"
            last_transition_time = time[i].item()

    n_fold = len(folding_times)
    n_unfold = len(unfolding_times)

    mean_fold_time = sum(folding_times) / n_fold if n_fold > 0 else float("inf")
    mean_unfold_time = sum(unfolding_times) / n_unfold if n_unfold > 0 else float("inf")

    return {
        "n_folding_events": n_fold,
        "n_unfolding_events": n_unfold,
        "mean_folding_time": mean_fold_time,
        "mean_unfolding_time": mean_unfold_time,
        "folding_rate": 1.0 / mean_fold_time if mean_fold_time < float("inf") else 0.0,
        "unfolding_rate": 1.0 / mean_unfold_time if mean_unfold_time < float("inf") else 0.0,
    }


def label_states(
    rmsd_trace: torch.Tensor,
    folded_cutoff: float = 0.2,
    unfolded_cutoff: float = 0.6,
) -> torch.Tensor:
    """Label each frame as folded (1), unfolded (0), or transition (-1).

    Parameters
    ----------
    rmsd_trace : torch.Tensor, shape (n_frames,)
    folded_cutoff, unfolded_cutoff : float

    Returns
    -------
    torch.Tensor, shape (n_frames,), dtype long
        1 = folded, 0 = unfolded, -1 = intermediate.
    """
    labels = torch.full_like(rmsd_trace, -1, dtype=torch.long)
    labels[rmsd_trace < folded_cutoff] = 1
    labels[rmsd_trace > unfolded_cutoff] = 0
    return labels
