r"""Finite-temperature string (FTS) method (Paper 1).

The FTS method maintains a discretized path (string) through configuration
space, with nodes phi^alpha.  Sampling is confined to Voronoi cells around
each node.  The path evolves to minimize the mean distance to sampled
configurations within each cell.

Path update (gradient descent on the FTS objective):

.. math::
    \varphi^\alpha_{k+1} = \varphi^\alpha_k - \eta
        \left(\varphi^\alpha_k - \langle x \rangle_{\mathcal{R}^\alpha_k}\right)
        + \lambda_S \nabla_{\varphi^\alpha}
          \sum_\alpha |\varphi^{\alpha+1} - \varphi^\alpha|^2

followed by equal-arc-length reparametrization.

References
----------
.. [1] Hasyim, Batton, Mandadapu. arXiv:2107.13522, Section 3.2.
.. [2] Vanden-Eijnden, Venturoli. J. Chem. Phys. 130, 194103 (2009).
"""

from __future__ import annotations

import torch


class FTSString:
    """Finite-temperature string represented as a sequence of nodes.

    Parameters
    ----------
    nodes : torch.Tensor, shape (n_nodes, dim)
        Initial string node positions.
    smoothing : float
        Smoothing penalty lambda_S for the arc-length regularizer.
    """

    def __init__(self, nodes: torch.Tensor, smoothing: float = 0.1) -> None:
        self.nodes = nodes.clone()
        self.smoothing = smoothing
        self.n_nodes = nodes.shape[0]
        self.dim = nodes.shape[1]

    def voronoi_assignment(self, x: torch.Tensor) -> torch.Tensor:
        """Assign each configuration to the nearest string node (Voronoi cell).

        Parameters
        ----------
        x : torch.Tensor, shape (batch, dim)

        Returns
        -------
        torch.Tensor, shape (batch,), dtype long
            Index of the nearest node for each configuration.
        """
        dists = torch.cdist(x, self.nodes)
        return dists.argmin(dim=1)

    def update(
        self,
        samples_per_cell: dict[int, torch.Tensor],
        learning_rate: float = 0.01,
    ) -> None:
        """Update string nodes toward cell means with smoothing.

        Parameters
        ----------
        samples_per_cell : dict[int, Tensor]
            Maps cell index -> (n_samples, dim) tensor of configurations.
        learning_rate : float
            Step size eta for the gradient descent.
        """
        for alpha in range(self.n_nodes):
            if alpha not in samples_per_cell or len(samples_per_cell[alpha]) == 0:
                continue
            cell_mean = samples_per_cell[alpha].mean(dim=0)
            mean_shift = self.nodes[alpha] - cell_mean

            smooth_grad = torch.zeros(self.dim, dtype=self.nodes.dtype, device=self.nodes.device)
            if alpha > 0:
                smooth_grad -= self.nodes[alpha - 1] - self.nodes[alpha]
            if alpha < self.n_nodes - 1:
                smooth_grad -= self.nodes[alpha + 1] - self.nodes[alpha]
            smooth_grad *= self.smoothing

            self.nodes[alpha] -= learning_rate * (mean_shift + smooth_grad)

        self._reparametrize()

    def _reparametrize(self) -> None:
        """Enforce equal arc-length spacing along the string."""
        seg_lengths = torch.norm(
            self.nodes[1:] - self.nodes[:-1], dim=1
        )
        cumulative = torch.cat([torch.zeros(1, device=seg_lengths.device), seg_lengths.cumsum(0)])
        total_length = cumulative[-1]
        if total_length < 1e-12:
            return

        target_s = torch.linspace(0, total_length.item(), self.n_nodes, device=self.nodes.device)

        new_nodes = torch.zeros_like(self.nodes)
        new_nodes[0] = self.nodes[0].clone()
        new_nodes[-1] = self.nodes[-1].clone()

        for i in range(1, self.n_nodes - 1):
            idx = torch.searchsorted(cumulative, target_s[i]).clamp(1, self.n_nodes - 1).item()
            frac = (target_s[i] - cumulative[int(idx) - 1]) / (
                cumulative[int(idx)] - cumulative[int(idx) - 1] + 1e-30
            )
            new_nodes[i] = (1 - frac) * self.nodes[int(idx) - 1] + frac * self.nodes[int(idx)]

        self.nodes = new_nodes


class FTSPathBias:
    """Path-based umbrella bias after FTS convergence (Paper 1, Eq. umbrella-fts).

    Anisotropic harmonic wells centered on string nodes with different
    spring constants parallel and perpendicular to the string.

    Parameters
    ----------
    string : FTSString
        Converged string.
    kappa_parallel : float
        Spring constant along the string tangent.
    kappa_perp : float
        Spring constant perpendicular to the string (softer).
    """

    def __init__(
        self,
        string: FTSString,
        kappa_parallel: float = 100.0,
        kappa_perp: float = 10.0,
    ) -> None:
        self.string = string
        self.kappa_parallel = kappa_parallel
        self.kappa_perp = kappa_perp

    def bias_energy(self, x: torch.Tensor, window_idx: int) -> torch.Tensor:
        """Compute the path-based umbrella bias for a given window.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, dim)
        window_idx : int
            Which string node to bias toward.

        Returns
        -------
        torch.Tensor, shape (batch,)
        """
        node = self.string.nodes[window_idx]
        dx = x - node.unsqueeze(0)

        tangent = self._tangent(window_idx)
        proj_parallel = (dx * tangent.unsqueeze(0)).sum(dim=-1, keepdim=True) * tangent.unsqueeze(0)
        proj_perp = dx - proj_parallel

        e_par = 0.5 * self.kappa_parallel * (proj_parallel**2).sum(dim=-1)
        e_perp = 0.5 * self.kappa_perp * (proj_perp**2).sum(dim=-1)
        return e_par + e_perp

    def _tangent(self, idx: int) -> torch.Tensor:
        """Unit tangent vector at node idx."""
        nodes = self.string.nodes
        if idx == 0:
            t = nodes[1] - nodes[0]
        elif idx == self.string.n_nodes - 1:
            t = nodes[-1] - nodes[-2]
        else:
            t = nodes[idx + 1] - nodes[idx - 1]
        return t / (t.norm() + 1e-30)
