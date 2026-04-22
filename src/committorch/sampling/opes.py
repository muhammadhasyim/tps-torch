r"""On-the-fly Probability Enhanced Sampling (OPES) on z(x).

Faithful reimplementation of the OPES_METAD algorithm from PLUMED
(Invernizzi & Parrinello, J. Phys. Chem. Lett. 2020).

Unlike well-tempered metadynamics which sums Gaussian hills, OPES
constructs a bias from a kernel density estimate (KDE) of the
*unbiased* distribution P(s), properly accounting for the bias
during estimation via reweighting:

.. math::
    V_n(s) = \frac{1-1/\gamma}{\beta}
             \log\!\left(\frac{\hat{P}_n(s)}{Z_n} + \varepsilon\right)

where:
- gamma is the bias factor (effective temperature T_eff = gamma * T),
- hat{P}_n(s) is the reweighted KDE of the unbiased distribution,
- Z_n normalizes over the explored CV space,
- epsilon regularizes unexplored regions.

Key algorithmic features matching PLUMED:
1. Kernels are reweighted: each sample contributes weight
   w_i = exp(beta * V(s_i)) to the KDE.
2. Kernel compression: nearby kernels (within threshold * sigma)
   are merged to prevent unbounded growth.
3. The bias becomes quasi-static once the KDE converges.

References
----------
.. [1] Invernizzi, Parrinello. J. Phys. Chem. Lett. 11, 2731 (2020).
.. [2] Trizio, Kang, Parrinello. arXiv:2410.17029, Methods + SI.
.. [3] PLUMED source: src/opes/OPESmetad.cpp
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..models.base import CommittorModel


@dataclass
class _Kernel:
    """A single Gaussian kernel in the compressed KDE."""

    center: float
    sigma: float
    weight: float


class OPESBias:
    r"""OPES_METAD bias on the latent z(x), following PLUMED's algorithm.

    The bias is constructed from a reweighted KDE of the unbiased
    distribution in z-space, with kernel compression.

    Parameters
    ----------
    model : CommittorModel
        Committor network (provides z(x)).
    sigma : float
        Width of Gaussian kernels in z-space.
    barrier : float
        Expected free energy barrier (in kT units).
        Used to set default epsilon and bias_factor if not provided.
    bias_factor : float or None
        Well-tempering bias factor gamma (>1).
        If None, set to barrier (matching PLUMED's default).
    beta : float
        Inverse temperature 1/(k_B T).
    epsilon : float or None
        Regularization constant.  If None, set from barrier:
        epsilon = exp(-barrier / (1 - 1/gamma)).
    compression_threshold : float
        Merge kernels closer than this (in units of sigma).
        Set to 0 to disable compression.
    """

    def __init__(
        self,
        model: CommittorModel,
        sigma: float = 0.5,
        barrier: float = 5.0,
        bias_factor: float | None = None,
        beta: float = 1.0,
        epsilon: float | None = None,
        compression_threshold: float = 1.0,
    ) -> None:
        self.model = model
        self.sigma = sigma
        self.beta = beta
        kT = 1.0 / beta

        if bias_factor is None:
            self.bias_factor = barrier  # PLUMED default: gamma = barrier/kT
        else:
            self.bias_factor = bias_factor

        self.bias_prefactor = 1.0 - 1.0 / self.bias_factor

        if epsilon is None:
            self.epsilon = max(
                torch.exp(torch.tensor(-barrier / self.bias_prefactor)).item(),
                1e-10,
            )
        else:
            self.epsilon = epsilon

        self.compression_threshold2 = compression_threshold**2
        self._kernels: list[_Kernel] = []
        self._sum_weights: float = self.epsilon**self.bias_prefactor
        self._sum_weights2: float = self._sum_weights**2
        self._counter: int = 1
        self._Zed: float = 1.0

    @property
    def n_kernels(self) -> int:
        """Number of compressed kernels."""
        return len(self._kernels)

    @property
    def n_hills(self) -> int:
        """Alias for n_kernels, backward compat."""
        return self.n_kernels

    def _kde_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluate the KDE probability estimate at z values.

        Parameters
        ----------
        z : torch.Tensor, shape (batch,)

        Returns
        -------
        torch.Tensor, shape (batch,)
            Unnormalized probability P_n(s).
        """
        if not self._kernels:
            return torch.zeros_like(z)

        centers = torch.tensor(
            [k.center for k in self._kernels], dtype=z.dtype, device=z.device
        )
        sigmas = torch.tensor(
            [k.sigma for k in self._kernels], dtype=z.dtype, device=z.device
        )
        weights = torch.tensor(
            [k.weight for k in self._kernels], dtype=z.dtype, device=z.device
        )

        # (batch, n_kernels)
        dz = z.unsqueeze(-1) - centers.unsqueeze(0)
        gaussians = torch.exp(-0.5 * (dz / sigmas.unsqueeze(0)) ** 2)
        # Normalize each Gaussian: 1/(sqrt(2*pi)*sigma)
        norm = 1.0 / (sigmas.unsqueeze(0) * torch.sqrt(torch.tensor(2.0 * torch.pi)))
        weighted_gaussians = weights.unsqueeze(0) * norm * gaussians

        prob = weighted_gaussians.sum(dim=-1)

        # Normalize by the KDE norm (sum_weights for OPES_METAD)
        kde_norm = self._sum_weights
        return prob / (kde_norm + 1e-30)

    def _compute_bias(self, z: torch.Tensor) -> torch.Tensor:
        """Compute V(z) = (1-1/gamma)/beta * log(P(z)/Z + epsilon).

        Parameters
        ----------
        z : torch.Tensor, shape (batch,)

        Returns
        -------
        torch.Tensor, shape (batch,)
        """
        prob = self._kde_prob(z)
        kT = 1.0 / self.beta
        return kT * self.bias_prefactor * torch.log(prob / self._Zed + self.epsilon)

    def deposit_kernel(self, z_val: float) -> None:
        """Deposit a new kernel at the given z value.

        Following PLUMED: the kernel's weight is the reweighting factor
        w = exp(beta * V(z_val)), which corrects the biased distribution
        back to the unbiased one.

        Parameters
        ----------
        z_val : float
            Current value of the collective variable z.
        """
        # Compute reweighting factor for this sample
        current_bias = self._compute_bias(torch.tensor([z_val])).item()
        weight = torch.exp(torch.tensor(self.beta * current_bias)).item()

        # Update running sums
        self._sum_weights += weight
        self._sum_weights2 += weight**2
        self._counter += 1

        new_kernel = _Kernel(center=z_val, sigma=self.sigma, weight=weight)

        # Try to merge with existing kernel (compression)
        merged = False
        if self.compression_threshold2 > 0:
            merged = self._try_merge(new_kernel)

        if not merged:
            self._kernels.append(new_kernel)

        # Update Z_n (normalization over explored CV space)
        self._update_zed()

    def deposit_hill(self, z_val: float) -> None:
        """Alias for deposit_kernel (backward compatibility)."""
        self.deposit_kernel(z_val)

    def _try_merge(self, new_kernel: _Kernel) -> bool:
        """Try to merge the new kernel with the closest existing one.

        Merges if the squared distance (in sigma units) is below
        compression_threshold^2.

        Parameters
        ----------
        new_kernel : _Kernel

        Returns
        -------
        bool
            True if merged, False if no merge happened.
        """
        if not self._kernels:
            return False

        best_idx = -1
        best_dist2 = float("inf")

        for i, k in enumerate(self._kernels):
            avg_sigma = 0.5 * (k.sigma + new_kernel.sigma)
            dist2 = ((k.center - new_kernel.center) / avg_sigma) ** 2
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_idx = i

        if best_dist2 < self.compression_threshold2:
            old = self._kernels[best_idx]
            total_w = old.weight + new_kernel.weight
            # Weight-averaged center and sigma
            merged_center = (old.weight * old.center + new_kernel.weight * new_kernel.center) / total_w
            merged_sigma = (
                (old.weight * old.sigma**2 + new_kernel.weight * new_kernel.sigma**2)
                / total_w
            ) ** 0.5
            self._kernels[best_idx] = _Kernel(
                center=merged_center, sigma=merged_sigma, weight=total_w
            )
            return True

        return False

    def _update_zed(self) -> None:
        """Update Z_n: average probability over all kernel centers."""
        if not self._kernels:
            self._Zed = 1.0
            return

        centers = torch.tensor([k.center for k in self._kernels])
        prob = self._kde_prob(centers)
        kde_norm = self._sum_weights
        self._Zed = prob.mean().item()

    def bias_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the OPES bias energy at configurations x.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, dim)

        Returns
        -------
        torch.Tensor, shape (batch,)
        """
        z = self.model.latent(x)
        if not self._kernels:
            kT = 1.0 / self.beta
            return torch.full_like(z, kT * self.bias_prefactor * torch.log(
                torch.tensor(self.epsilon)
            ).item())
        return self._compute_bias(z)

    def as_bias_fn(self):
        """Return a callable suitable for ``Simulator.set_bias``."""
        return self.bias_energy

    @property
    def effective_sample_size(self) -> float:
        """Kish effective sample size: (sum w)^2 / (sum w^2)."""
        return self._sum_weights**2 / (self._sum_weights2 + 1e-30)

    @property
    def rct(self) -> float:
        r"""Estimate of c(t) = (1/beta) * log(<exp(beta*V)>).

        Should converge to a fixed value as the bias converges.
        """
        kT = 1.0 / self.beta
        return kT * torch.log(torch.tensor(self._sum_weights / self._counter)).item()


class CombinedVKOPES:
    """Combined V_K + OPES bias (Paper 2's V_eff = V_K + V_OPES).

    Parameters
    ----------
    vk : KolmogorovBias instance
    opes : OPESBias instance
    """

    def __init__(self, vk, opes: OPESBias) -> None:
        self.vk = vk
        self.opes = opes

    def bias_energy(self, x: torch.Tensor) -> torch.Tensor:
        return self.vk.bias_energy(x) + self.opes.bias_energy(x)

    def as_bias_fn(self):
        return self.bias_energy

    def reweighting_factor(
        self, x: torch.Tensor, beta: float
    ) -> torch.Tensor:
        r"""Compute importance weights w_i = exp(beta * V_eff(x_i)).

        Normalized so that sum(w_i) = 1.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, dim)
        beta : float

        Returns
        -------
        torch.Tensor, shape (batch,)
            Normalized weights.
        """
        v_eff = self.bias_energy(x)
        log_w = beta * v_eff
        log_w = log_w - log_w.max()
        w = torch.exp(log_w)
        return w / w.sum()
