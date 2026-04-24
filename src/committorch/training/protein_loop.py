"""Protein-scale active learning loops using OpenMM for dynamics.

These loops implement the feedback cycle for training committor models
on real molecular systems:

1. Biased MD with OpenMM (using TorchForce for committor-derived biases)
2. Collect configurations + reweighting factors
3. Train the committor (BKE + boundary + optional supervised loss)
4. Re-serialize model, swap TorchForce, reinitialize OpenMM context
5. Repeat

Two variants:
- ``ProteinOPESLoop``: V_K + OPES metadynamics (Paper 2)
- ``ProteinUmbrellaLoop``: committor-umbrella windows (Paper 1)
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from ..models.base import CommittorModel
from ..simulators.openmm_sim import OpenMMSimulator
from .loop import ActiveLearningLoop, LoopState

logger = logging.getLogger(__name__)


class ProteinOPESLoop(ActiveLearningLoop):
    """Active learning with V_K + OPES for protein systems via OpenMM.

    The key difference from the toy ``OPESLoop``:
    - Uses ``OpenMMSimulator`` with TorchForce-based biasing
    - After each training epoch, re-serializes the committor model and
      calls ``simulator.refresh_bias()`` to inject updated parameters
    - Handles 3D atomic positions ``(n_atoms, 3)``
    - Boundary configurations come from PDB structures

    Parameters
    ----------
    model : CommittorModel
        Committor model (MLP, MACE, etc.).
    simulator : OpenMMSimulator
        OpenMM simulation with TorchForce bias.
    x_a : torch.Tensor
        Folded (reactant) configurations, shape (n_configs, n_atoms*3).
    x_b : torch.Tensor
        Unfolded (product) configurations, shape (n_configs, n_atoms*3).
    vk_lambda : float
        Kolmogorov bias strength.
    opes_sigma : float
        OPES kernel width in z-space.
    opes_barrier : float
        Expected free energy barrier in kT.
    opes_bias_factor : float
        Well-tempering factor gamma.
    opes_compression_threshold : float
        Merge OPES kernels closer than this.
    kernel_deposit_interval : int
        Deposit a kernel every this many steps.
    collect_interval : int
        Collect positions every this many steps during MD.
    bias_type : str
        Type of bias for TorchForce: "umbrella", "kolmogorov", "combined_vk_opes".
    bias_params : dict or None
        Parameters for the TorchForce bias module.
    """

    def __init__(
        self,
        model: CommittorModel,
        simulator: OpenMMSimulator,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        vk_lambda: float = 1.0,
        opes_sigma: float = 0.5,
        opes_barrier: float = 5.0,
        opes_bias_factor: float = 10.0,
        opes_compression_threshold: float = 1.0,
        kernel_deposit_interval: int = 100,
        collect_interval: int = 50,
        bias_type: str = "kolmogorov",
        bias_params: dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, simulator, x_a, x_b, **kwargs)

        from ..sampling.kolmogorov import KolmogorovBias
        from ..sampling.opes import OPESBias

        self.openmm_sim: OpenMMSimulator = simulator
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
        self.kernel_deposit_interval = kernel_deposit_interval
        self.collect_interval = collect_interval
        self._bias_type = bias_type
        self._bias_params = dict(bias_params) if bias_params else {}
        self._bias_params.setdefault("lambda", vk_lambda)
        self._bias_params.setdefault("beta", beta)

        self.openmm_sim.model = model

        if self.openmm_sim.simulation is not None:
            self._setup_openmm_bias()

    def _setup_openmm_bias(self) -> None:
        """Initialize the TorchForce bias in the OpenMM system.

        Silently skips if openmm-torch is not available (e.g. unit tests).
        """
        try:
            self.openmm_sim.add_committor_bias(
                bias_type=self._bias_type,
                bias_params=self._bias_params,
            )
        except ImportError:
            logger.warning(
                "openmm-torch not available; skipping TorchForce setup"
            )

    def collect_samples(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run biased OpenMM MD and collect snapshots.

        Returns
        -------
        configs : torch.Tensor, shape (n_snapshots, dim)
        weights : torch.Tensor or None
        """
        configs = self.openmm_sim.step_and_collect(
            n_steps=self.n_sim_steps,
            collect_interval=self.collect_interval,
        )
        if self.openmm_sim.flatten_positions and configs.dim() == 3:
            configs = configs.reshape(configs.shape[0], -1)

        with torch.no_grad():
            for i in range(configs.shape[0]):
                if (i + 1) % self.kernel_deposit_interval == 0:
                    z_val = self.model.latent(configs[i:i+1]).item()
                    self.opes.deposit_kernel(z_val)

        logger.info(
            "Collected %d snapshots, %d OPES kernels total",
            configs.shape[0],
            self.opes.n_kernels,
        )
        return configs, None

    def update_bias(self) -> None:
        """Re-serialize model and swap TorchForce in OpenMM."""
        try:
            self.openmm_sim.refresh_bias(self.model)
            logger.debug("Refreshed OpenMM bias after training step")
        except ImportError:
            logger.warning("openmm-torch not available; skipping bias refresh")

    @property
    def diagnostics(self) -> dict[str, float]:
        """Return OPES convergence diagnostics."""
        return {
            "n_kernels": self.opes.n_kernels,
            "rct": self.opes.rct,
            "ess": self.opes.effective_sample_size,
        }


class ProteinUmbrellaLoop(ActiveLearningLoop):
    """Active learning with committor-umbrella windows for proteins.

    Runs M replicas in parallel (conceptually), each biased toward a
    different committor target q_i.  After training, the umbrella biases
    automatically reflect updated model parameters via ``refresh_bias()``.

    Parameters
    ----------
    model : CommittorModel
        Committor model.
    simulator : OpenMMSimulator
        Template OpenMM simulation (will be conceptually replicated).
    x_a : torch.Tensor
        Folded boundary configs.
    x_b : torch.Tensor
        Unfolded boundary configs.
    n_windows : int
        Number of umbrella windows.
    spring_constant : float
        Umbrella spring constant in kJ/mol.
    """

    def __init__(
        self,
        model: CommittorModel,
        simulator: OpenMMSimulator,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        n_windows: int = 10,
        spring_constant: float = 100.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, simulator, x_a, x_b, **kwargs)
        self.openmm_sim: OpenMMSimulator = simulator
        self.openmm_sim.model = model
        self.n_windows = n_windows
        self.spring_constant = spring_constant
        self.targets = torch.linspace(0.0, 1.0, n_windows)

    def collect_samples(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run umbrella-biased MD for each window and collect configs.

        For each window i, sets up umbrella bias with target q_i,
        runs MD, and collects final positions.
        """
        all_configs: list[torch.Tensor] = []

        for i in range(self.n_windows):
            target = self.targets[i].item()
            try:
                self.openmm_sim.add_committor_bias(
                    bias_type="umbrella",
                    bias_params={
                        "target": target,
                        "spring_constant": self.spring_constant,
                    },
                )
            except ImportError:
                logger.warning("openmm-torch not available; running unbiased")

            configs = self.openmm_sim.step_and_collect(
                n_steps=self.n_sim_steps,
                collect_interval=self.n_sim_steps,
            )
            all_configs.append(configs)

        all_configs_t = torch.cat(all_configs, dim=0)
        if self.openmm_sim.flatten_positions and all_configs_t.dim() == 3:
            all_configs_t = all_configs_t.reshape(all_configs_t.shape[0], -1)

        logger.info(
            "Collected %d configs from %d umbrella windows",
            all_configs_t.shape[0],
            self.n_windows,
        )
        return all_configs_t, None

    def update_bias(self) -> None:
        """Re-serialize model to update umbrella biases."""
        try:
            self.openmm_sim.refresh_bias(self.model)
        except ImportError:
            logger.warning("openmm-torch not available; skipping bias refresh")
