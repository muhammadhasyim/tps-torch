"""Active learning loop: orchestrates sample <-> train iterations.

The loop alternates between:
1. Running biased simulation to collect configurations
2. Computing the BKE + boundary + supervised losses
3. Updating the committor model parameters
4. (Optionally) updating the bias / string

This module provides an abstract ``ActiveLearningLoop`` and concrete
subclasses for different enhanced sampling methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.optim as optim
from tqdm import trange

from ..models.base import CommittorModel
from ..simulators.base import Simulator
from ..training.losses import BKELoss, BoundaryLoss


@dataclass
class LoopState:
    """Mutable state tracked across training iterations."""

    iteration: int = 0
    loss_history: list[float] = field(default_factory=list)
    bke_history: list[float] = field(default_factory=list)
    boundary_history: list[float] = field(default_factory=list)


class ActiveLearningLoop(ABC):
    """Abstract base for the sample-train feedback loop.

    Parameters
    ----------
    model : CommittorModel
        Committor network to train.
    simulator : Simulator
        Dynamics engine.
    x_a : torch.Tensor
        Configurations in reactant basin A.
    x_b : torch.Tensor
        Configurations in product basin B.
    lr : float
        Learning rate for the optimizer.
    lambda_boundary : float
        Weight for boundary loss.
    n_sim_steps : int
        Simulation steps per data collection round.
    batch_size : int
        Number of configurations per batch.
    """

    def __init__(
        self,
        model: CommittorModel,
        simulator: Simulator,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        lr: float = 1e-3,
        lambda_boundary: float = 1.0,
        n_sim_steps: int = 100,
        batch_size: int = 256,
    ) -> None:
        self.model = model
        self.simulator = simulator
        self.x_a = x_a
        self.x_b = x_b
        self.lambda_boundary = lambda_boundary
        self.n_sim_steps = n_sim_steps
        self.batch_size = batch_size

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.bke_loss_fn = BKELoss(model)
        self.boundary_loss_fn = BoundaryLoss(model)
        self.state = LoopState()

    @abstractmethod
    def collect_samples(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run biased simulation and collect samples.

        Returns
        -------
        configs : torch.Tensor, shape (batch, dim)
            Sampled configurations.
        weights : torch.Tensor or None, shape (batch,)
            Importance weights for BKE loss.
        """

    @abstractmethod
    def update_bias(self) -> None:
        """Update bias potential / string after model parameter update."""

    def train_step(self, configs: torch.Tensor, weights: torch.Tensor | None) -> float:
        """One gradient update on BKE + boundary loss.

        Uses per-chunk gradient accumulation for BKE to avoid holding
        all MACE computation graphs in GPU memory simultaneously.
        """
        self.optimizer.zero_grad()
        bke = self.bke_loss_fn(configs, weights, accumulate_grads=True)
        bc = self.boundary_loss_fn(self.x_a, self.x_b)
        bc_scaled = self.lambda_boundary * bc
        bc_scaled.backward()
        self.optimizer.step()

        bke_val = bke.item()
        bc_val = bc.item()
        loss_val = bke_val + self.lambda_boundary * bc_val
        self.state.bke_history.append(bke_val)
        self.state.boundary_history.append(bc_val)
        self.state.loss_history.append(loss_val)
        return loss_val

    def run(self, n_iterations: int, progress: bool = True) -> LoopState:
        """Run the full active learning loop.

        Parameters
        ----------
        n_iterations : int
            Number of sample-train iterations.
        progress : bool
            Show progress bar.

        Returns
        -------
        LoopState
            Training history.
        """
        iterator = trange(n_iterations, desc="Training", disable=not progress)
        for _ in iterator:
            configs, weights = self.collect_samples()
            loss_val = self.train_step(configs, weights)
            self.update_bias()
            self.state.iteration += 1
            iterator.set_postfix(loss=f"{loss_val:.4e}")
        return self.state


class OPESLoop(ActiveLearningLoop):
    """Training loop with V_K + OPES metadynamics (Paper 2).

    Self-consistent iteration:
    1. Train committor on current dataset
    2. Run biased MD with V_K + OPES on z(x)
    3. Reweight samples, append to dataset
    4. Repeat

    Uses the PLUMED-style OPES_METAD algorithm: KDE-based bias with
    kernel compression and proper reweighting.

    Parameters
    ----------
    vk_lambda : float
        Kolmogorov bias strength.
    opes_sigma : float
        OPES kernel width in z-space.
    opes_barrier : float
        Expected free energy barrier (in kT units).
    opes_bias_factor : float
        Well-tempering factor gamma.
    opes_compression_threshold : float
        Merge kernels closer than this (in sigma units).
    kernel_deposit_interval : int
        Deposit a kernel every this many simulation steps.
    """

    def __init__(
        self,
        model: CommittorModel,
        simulator: Simulator,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        vk_lambda: float = 1.0,
        opes_sigma: float = 0.5,
        opes_barrier: float = 5.0,
        opes_bias_factor: float = 10.0,
        opes_compression_threshold: float = 1.0,
        kernel_deposit_interval: int = 10,
        # Legacy aliases
        opes_height: float | None = None,
        hill_deposit_interval: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(model, simulator, x_a, x_b, **kwargs)

        from ..sampling.kolmogorov import KolmogorovBias
        from ..sampling.opes import OPESBias, CombinedVKOPES

        if hill_deposit_interval is not None:
            kernel_deposit_interval = hill_deposit_interval

        beta = 1.0 / simulator.kT
        self.vk = KolmogorovBias(model, lam=vk_lambda, beta=beta)
        self.opes = OPESBias(
            model,
            sigma=opes_sigma,
            barrier=opes_barrier,
            bias_factor=opes_bias_factor,
            beta=beta,
            compression_threshold=opes_compression_threshold,
        )
        self.combined = CombinedVKOPES(self.vk, self.opes)
        self.kernel_deposit_interval = kernel_deposit_interval
        self._current_position: torch.Tensor | None = None

    def collect_samples(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._current_position is None:
            x_start = self.x_a.mean(dim=0)
            x_end = self.x_b.mean(dim=0)
            self._current_position = 0.5 * (x_start + x_end)
            self._current_position = self._current_position.unsqueeze(0)

        self.simulator.set_bias(self.combined.as_bias_fn())
        configs_list = []

        pos = self._current_position
        for step_i in range(self.n_sim_steps):
            pos = self.simulator.step(pos, n_steps=1)
            configs_list.append(pos.detach().clone())

            if (step_i + 1) % self.kernel_deposit_interval == 0:
                with torch.no_grad():
                    z_val = self.model.latent(pos).item()
                self.opes.deposit_kernel(z_val)

        self._current_position = pos.detach()
        self.simulator.set_bias(None)

        configs = torch.cat(configs_list, dim=0)
        beta = 1.0 / self.simulator.kT
        weights = self.combined.reweighting_factor(configs, beta)
        return configs, weights

    def update_bias(self) -> None:
        pass


class UmbrellaLoop(ActiveLearningLoop):
    """Training loop with committor-based umbrella sampling (Paper 1).

    Runs M replica simulations, each biased toward a different committor
    target.  After each training step, the umbrella biases automatically
    reflect the updated model parameters since they depend on q(x; theta).

    Parameters
    ----------
    n_windows : int
        Number of umbrella windows.
    spring_constant : float
        Umbrella spring constant.
    """

    def __init__(
        self,
        model: CommittorModel,
        simulator: Simulator,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        n_windows: int = 10,
        spring_constant: float = 100.0,
        **kwargs,
    ) -> None:
        super().__init__(model, simulator, x_a, x_b, **kwargs)
        self.n_windows = n_windows
        self.spring_constant = spring_constant

        self.targets = torch.linspace(0.0, 1.0, n_windows)
        self._replica_positions: torch.Tensor | None = None

    def _init_replicas(self) -> None:
        """Initialize replica positions by interpolation between A and B."""
        x_start = self.x_a.mean(dim=0)
        x_end = self.x_b.mean(dim=0)
        fracs = self.targets.unsqueeze(1)
        self._replica_positions = (1 - fracs) * x_start + fracs * x_end

    def collect_samples(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._replica_positions is None:
            self._init_replicas()

        all_configs = []
        assert self._replica_positions is not None
        for i in range(self.n_windows):
            target = self.targets[i].item()

            def bias_fn(x, t=target):
                q = self.model(x)
                return 0.5 * self.spring_constant * (q - t) ** 2

            self.simulator.set_bias(bias_fn)
            pos = self._replica_positions[i].unsqueeze(0)
            pos = self.simulator.step(pos, n_steps=self.n_sim_steps)
            self._replica_positions[i] = pos.squeeze(0).detach()
            all_configs.append(pos.detach())

        self.simulator.set_bias(None)
        return torch.cat(all_configs, dim=0), None

    def update_bias(self) -> None:
        pass
