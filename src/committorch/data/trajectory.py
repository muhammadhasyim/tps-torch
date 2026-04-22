"""Trajectory loading and saving utilities.

Wraps MDTraj for reading common trajectory formats (.dcd, .xtc, .trr, .h5)
and converting to PyTorch tensors.

Also provides a lightweight trajectory container for use without MDTraj.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch

try:
    import mdtraj
    HAS_MDTRAJ = True
except ImportError:
    HAS_MDTRAJ = False


@dataclass
class Trajectory:
    """Lightweight trajectory container.

    Parameters
    ----------
    positions : torch.Tensor, shape (n_frames, n_atoms, 3)
        Atomic positions in nm.
    time : torch.Tensor or None, shape (n_frames,)
        Simulation time for each frame.
    topology : any
        Optional topology information (MDTraj topology, etc.).
    metadata : dict
        Arbitrary metadata.
    """

    positions: torch.Tensor
    time: torch.Tensor | None = None
    topology: object = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return self.positions.shape[0]

    @property
    def n_atoms(self) -> int:
        return self.positions.shape[1]

    def __len__(self) -> int:
        return self.n_frames

    def slice(self, start: int, stop: int) -> Trajectory:
        """Return a sub-trajectory."""
        return Trajectory(
            positions=self.positions[start:stop],
            time=self.time[start:stop] if self.time is not None else None,
            topology=self.topology,
            metadata=self.metadata,
        )


def load_trajectory(
    trajectory_path: str | Path,
    topology_path: str | Path | None = None,
    stride: int = 1,
    atom_indices: list[int] | None = None,
) -> Trajectory:
    """Load a trajectory from disk using MDTraj.

    Parameters
    ----------
    trajectory_path : str or Path
        Path to trajectory file (.dcd, .xtc, .trr, .h5, etc.).
    topology_path : str or Path or None
        Path to topology file (.pdb, .gro, etc.).  Required for
        formats that don't include topology.
    stride : int
        Load every stride-th frame.
    atom_indices : list of int or None
        Subset of atom indices to load.

    Returns
    -------
    Trajectory
    """
    if not HAS_MDTRAJ:
        raise ImportError(
            "MDTraj is required for trajectory loading. "
            "Install with: pip install mdtraj"
        )

    if topology_path is not None:
        traj = mdtraj.load(
            str(trajectory_path),
            top=str(topology_path),
            stride=stride,
            atom_indices=atom_indices,
        )
    else:
        traj = mdtraj.load(
            str(trajectory_path),
            stride=stride,
            atom_indices=atom_indices,
        )

    positions = torch.tensor(traj.xyz, dtype=torch.float64)
    time = torch.tensor(traj.time, dtype=torch.float64) if traj.time is not None else None

    return Trajectory(
        positions=positions,
        time=time,
        topology=traj.topology,
        metadata={"source": str(trajectory_path)},
    )


def save_trajectory(
    traj: Trajectory,
    path: str | Path,
    topology_path: str | Path | None = None,
) -> None:
    """Save a trajectory to disk.

    Parameters
    ----------
    traj : Trajectory
    path : str or Path
        Output file path.
    topology_path : str or Path or None
        Topology file for formats that need it.
    """
    if not HAS_MDTRAJ:
        torch.save(
            {"positions": traj.positions, "time": traj.time, "metadata": traj.metadata},
            path,
        )
        return

    import numpy as np

    xyz = traj.positions.numpy()
    if traj.topology is not None:
        md_traj = mdtraj.Trajectory(
            xyz=xyz.astype(np.float32),
            topology=traj.topology,
        )
    else:
        n_atoms = xyz.shape[1]
        topology = mdtraj.Topology()
        chain = topology.add_chain()
        for i in range(n_atoms):
            res = topology.add_residue("UNK", chain)
            topology.add_atom("CA", mdtraj.element.carbon, res)
        md_traj = mdtraj.Trajectory(
            xyz=xyz.astype(np.float32),
            topology=topology,
        )

    md_traj.save(str(path))
