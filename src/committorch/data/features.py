"""Feature extraction from molecular configurations.

Provides routines for computing structural features commonly used
as inputs to committor models or for analysis.
"""

from __future__ import annotations

import torch


def pairwise_distances(
    positions: torch.Tensor,
    atom_pairs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute pairwise inter-atomic distances.

    Parameters
    ----------
    positions : torch.Tensor, shape (batch, n_atoms, 3) or (n_atoms, 3)
        Atomic positions.
    atom_pairs : torch.Tensor or None, shape (n_pairs, 2)
        Specific atom pairs.  If None, all pairs.

    Returns
    -------
    torch.Tensor
        Shape (batch, n_pairs) or (n_pairs,).
    """
    squeeze = positions.dim() == 2
    if squeeze:
        positions = positions.unsqueeze(0)

    if atom_pairs is None:
        n_atoms = positions.shape[1]
        idx = torch.triu_indices(n_atoms, n_atoms, offset=1)
        atom_pairs = torch.stack([idx[0], idx[1]], dim=1)

    pos_i = positions[:, atom_pairs[:, 0]]
    pos_j = positions[:, atom_pairs[:, 1]]
    dists = torch.sqrt(((pos_i - pos_j) ** 2).sum(dim=-1) + 1e-10)

    if squeeze:
        dists = dists.squeeze(0)
    return dists


def contact_map(
    positions: torch.Tensor,
    cutoff: float = 0.8,
    atom_pairs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary contact map: 1 if distance < cutoff, 0 otherwise.

    Parameters
    ----------
    positions : torch.Tensor
    cutoff : float
        Distance cutoff in nm.
    atom_pairs : torch.Tensor or None

    Returns
    -------
    torch.Tensor
        Same shape as pairwise_distances output, binary (0/1).
    """
    dists = pairwise_distances(positions, atom_pairs)
    return (dists < cutoff).float()


def rmsd(
    positions: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Root-mean-square deviation after optimal superposition (Kabsch).

    Simplified version without optimal rotation -- computes RMSD
    after centering both structures.

    Parameters
    ----------
    positions : torch.Tensor, shape (batch, n_atoms, 3) or (n_atoms, 3)
    reference : torch.Tensor, shape (n_atoms, 3)

    Returns
    -------
    torch.Tensor
        Shape (batch,) or scalar.
    """
    squeeze = positions.dim() == 2
    if squeeze:
        positions = positions.unsqueeze(0)

    ref = reference.unsqueeze(0)
    pos_centered = positions - positions.mean(dim=1, keepdim=True)
    ref_centered = ref - ref.mean(dim=1, keepdim=True)

    diff = pos_centered - ref_centered
    msd = (diff**2).sum(dim=-1).mean(dim=-1)
    result = torch.sqrt(msd)

    if squeeze:
        result = result.squeeze(0)
    return result


def calpha_distances(
    positions: torch.Tensor,
    ca_indices: list[int],
) -> torch.Tensor:
    """Pairwise distances between C-alpha atoms.

    Parameters
    ----------
    positions : torch.Tensor, shape (batch, n_atoms, 3)
    ca_indices : list of int
        Indices of C-alpha atoms in the full coordinate array.

    Returns
    -------
    torch.Tensor, shape (batch, n_ca_pairs)
    """
    ca_pos = positions[:, ca_indices]
    return pairwise_distances(ca_pos)
