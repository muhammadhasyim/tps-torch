"""Graph construction utilities for molecular GNN models.

Provides a ``build_graph`` function that takes raw positions and atomic
numbers and produces neighbor lists and edge features suitable for
MACE, SchNet, PaiNN, or other message-passing networks.

All operations are pure PyTorch (no external neighbor list libraries
required), making the graph construction differentiable and
JIT-traceable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GraphInput:
    """Container for molecular graph data.

    Attributes
    ----------
    positions : torch.Tensor, shape (n_atoms, 3)
        Atomic positions for a single configuration.
    atomic_numbers : torch.Tensor, shape (n_atoms,), dtype long
        Atomic numbers (e.g. 6 for carbon).
    edge_index : torch.Tensor, shape (2, n_edges), dtype long
        Source and destination indices for each edge.
    edge_vectors : torch.Tensor, shape (n_edges, 3)
        Displacement vectors r_j - r_i for each edge (i, j).
    edge_lengths : torch.Tensor, shape (n_edges,)
        ||r_j - r_i|| for each edge.
    batch_index : torch.Tensor, shape (n_atoms,), dtype long
        Batch assignment for each atom (0 for a single graph).
    n_atoms : int
        Total number of atoms.
    """
    positions: torch.Tensor
    atomic_numbers: torch.Tensor
    edge_index: torch.Tensor
    edge_vectors: torch.Tensor
    edge_lengths: torch.Tensor
    batch_index: torch.Tensor
    n_atoms: int


def build_graph(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cutoff: float = 5.0,
    self_interaction: bool = False,
) -> GraphInput:
    """Build a molecular graph from a single configuration.

    Parameters
    ----------
    positions : torch.Tensor, shape (n_atoms, 3)
        Atomic positions in angstroms.
    atomic_numbers : torch.Tensor, shape (n_atoms,)
        Atomic numbers.
    cutoff : float
        Neighbor cutoff distance in angstroms.
    self_interaction : bool
        If True, include self-loops (i == j).

    Returns
    -------
    GraphInput
    """
    n_atoms = positions.shape[0]

    # Pairwise displacement vectors: r_j - r_i
    diffs = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, 3)
    dists = torch.sqrt((diffs ** 2).sum(dim=-1) + 1e-12)    # (N, N)

    # Build mask for edges within cutoff
    mask = dists < cutoff
    if not self_interaction:
        mask = mask & ~torch.eye(n_atoms, dtype=torch.bool, device=positions.device)

    src, dst = torch.where(mask)
    edge_index = torch.stack([src, dst], dim=0)
    edge_vectors = diffs[src, dst]
    edge_lengths = dists[src, dst]

    batch_index = torch.zeros(n_atoms, dtype=torch.long, device=positions.device)

    return GraphInput(
        positions=positions,
        atomic_numbers=atomic_numbers.to(torch.long),
        edge_index=edge_index,
        edge_vectors=edge_vectors,
        edge_lengths=edge_lengths,
        batch_index=batch_index,
        n_atoms=n_atoms,
    )


def build_batch_graphs(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cutoff: float = 5.0,
) -> GraphInput:
    """Build batched molecular graphs from multiple configurations.

    Parameters
    ----------
    positions : torch.Tensor, shape (batch, n_atoms, 3)
        Atomic positions for each configuration.
    atomic_numbers : torch.Tensor, shape (n_atoms,)
        Shared atomic numbers across the batch.
    cutoff : float
        Neighbor cutoff distance.

    Returns
    -------
    GraphInput
        Batched graph with concatenated atoms and shifted edge indices.
    """
    batch_size = positions.shape[0]
    n_atoms_per = positions.shape[1]

    all_positions = []
    all_edge_index = []
    all_edge_vectors = []
    all_edge_lengths = []
    all_batch_index = []
    all_atomic_numbers = []

    offset = 0
    for b in range(batch_size):
        g = build_graph(positions[b], atomic_numbers, cutoff)
        all_positions.append(g.positions)
        all_edge_index.append(g.edge_index + offset)
        all_edge_vectors.append(g.edge_vectors)
        all_edge_lengths.append(g.edge_lengths)
        all_batch_index.append(
            torch.full((n_atoms_per,), b, dtype=torch.long, device=positions.device)
        )
        all_atomic_numbers.append(g.atomic_numbers)
        offset += n_atoms_per

    return GraphInput(
        positions=torch.cat(all_positions, dim=0),
        atomic_numbers=torch.cat(all_atomic_numbers, dim=0),
        edge_index=torch.cat(all_edge_index, dim=1),
        edge_vectors=torch.cat(all_edge_vectors, dim=0),
        edge_lengths=torch.cat(all_edge_lengths, dim=0),
        batch_index=torch.cat(all_batch_index, dim=0),
        n_atoms=offset,
    )


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    """Mean-pool node features by batch index.

    Parameters
    ----------
    src : torch.Tensor, shape (total_nodes, feat_dim)
    index : torch.Tensor, shape (total_nodes,)
        Batch assignment for each node.
    dim_size : int
        Number of graphs in the batch.

    Returns
    -------
    torch.Tensor, shape (dim_size, feat_dim)
    """
    out = torch.zeros(dim_size, src.shape[-1], dtype=src.dtype, device=src.device)
    count = torch.zeros(dim_size, 1, dtype=src.dtype, device=src.device)

    idx = index.unsqueeze(-1).expand_as(src)
    out.scatter_add_(0, idx, src)
    count.scatter_add_(0, index.unsqueeze(-1), torch.ones_like(index, dtype=src.dtype).unsqueeze(-1))
    return out / count.clamp(min=1)
