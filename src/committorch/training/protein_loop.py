"""Protein-scale active learning loops using OpenMM for dynamics.

These loops implement the feedback cycle for training committor models
on real molecular systems:

1. Biased MD with OpenMM (using TorchForce for committor-derived biases)
2. Collect configurations + reweighting factors
3. Train the committor (BKE + boundary + optional supervised loss)
4. Re-serialize model, swap TorchForce, reinitialize OpenMM context
5. Repeat

Two variants:
- ``ProteinOPESLoop``: V_K + OPES metadynamics (Paper 2, arXiv:2410.17029)
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
    r"""Active learning with V_K + OPES for protein systems via OpenMM.

    Implements the iterative scheme from arXiv:2410.17029:

    1. Run biased MD with V_eff = V_K + V_OPES (two TorchForce objects).
    2. Deposit OPES kernels on z(x) during MD (chunked: deposit then
       refresh the V_OPES TorchForce between chunks).
    3. Compute importance weights w_i = exp(beta V_eff(x_i)) / Z.
    4. Accumulate (configs, log-weights) across iterations.
    5. Train q_theta on the full accumulated dataset with reweighted BKE loss.
    6. Re-serialize model into TorchForce and repeat.

    Parameters
    ----------
    model : CommittorModel
        Committor model (MLP, MACE, etc.).
    simulator : OpenMMSimulator
        OpenMM simulation with TorchForce bias.
    x_a, x_b : torch.Tensor
        Boundary configurations, shape (n_configs, n_atoms*3).
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
        Deposit a kernel every this many collected snapshots.
    collect_interval : int
        Collect positions every this many MD steps.
    bias_type : str
        Type of bias for TorchForce.  ``"combined_vk_opes"`` (default)
        uses two TorchForces: V_K (kolmogorov) + V_OPES.
    bias_params : dict or None
        Parameters for the TorchForce bias module.
    max_buffer_size : int or None
        Cap for the accumulated dataset.  When exceeded, the oldest
        configs are dropped.  None means unlimited growth.
    atom_masses : torch.Tensor or None
        Atomic masses in amu, shape (n_atoms,).  Enables mass-weighted
        gradients in the BKE loss.  None falls back to unweighted.
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
        bias_type: str = "combined_vk_opes",
        bias_params: dict | None = None,
        max_buffer_size: int | None = None,
        atom_masses: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, simulator, x_a, x_b, **kwargs)

        from ..sampling.kolmogorov import KolmogorovBias
        from ..sampling.opes import CombinedVKOPES, OPESBias
        from .losses import BKELoss

        # Replace the base-class BKELoss with mass-weighted version
        if atom_masses is not None:
            self.bke_loss_fn = BKELoss(model, atom_masses=atom_masses)

        self.openmm_sim: OpenMMSimulator = simulator
        beta = 1.0 / simulator.kT
        self._beta = beta

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
        self.collect_interval = collect_interval
        self._bias_type = bias_type
        self._bias_params = dict(bias_params) if bias_params else {}
        self._bias_params.setdefault("lambda", vk_lambda)
        self._bias_params.setdefault("beta", beta)
        self.max_buffer_size = max_buffer_size

        # Accumulated dataset (paper: "all N_n configs until iteration n")
        self._config_buffer: list[torch.Tensor] = []
        self._log_weight_buffer: list[torch.Tensor] = []

        self.openmm_sim.model = model

        if self.openmm_sim.simulation is not None:
            self._setup_openmm_bias()

    def _setup_openmm_bias(self) -> None:
        """Initialize the TorchForce bias in the OpenMM system."""
        try:
            self.openmm_sim.add_committor_bias(
                bias_type=self._bias_type,
                bias_params=self._bias_params,
                opes_state=self.opes.state_dict(),
            )
        except ImportError:
            logger.warning(
                "openmm-torch not available; skipping TorchForce setup"
            )

    def collect_samples(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""Run biased OpenMM MD, deposit kernels, compute weights.

        MD is broken into chunks of ``kernel_deposit_interval`` snapshots.
        After each chunk, OPES kernels are deposited and the V_OPES
        TorchForce is refreshed so subsequent MD uses the updated bias.

        New configs are appended to the growing buffer.  Returns the
        *full accumulated* dataset with softmax-normalized importance
        weights:

        .. math::
            w_i = \frac{\exp(\beta\, V_{\mathrm{eff}}(x_i))}
                       {\sum_j \exp(\beta\, V_{\mathrm{eff}}(x_j))}

        Returns
        -------
        configs : torch.Tensor, shape (N_accumulated, dim)
        weights : torch.Tensor, shape (N_accumulated,)
        """
        total_snapshots = self.n_sim_steps // self.collect_interval
        if total_snapshots < 1:
            total_snapshots = 1

        chunk_size = max(self.kernel_deposit_interval, 1)
        n_chunks = max(total_snapshots // chunk_size, 1)
        steps_per_chunk = chunk_size * self.collect_interval

        new_configs_list: list[torch.Tensor] = []

        for _ in range(n_chunks):
            configs_chunk = self.openmm_sim.step_and_collect(
                n_steps=steps_per_chunk,
                collect_interval=self.collect_interval,
            )
            if self.openmm_sim.flatten_positions and configs_chunk.dim() == 3:
                configs_chunk = configs_chunk.reshape(configs_chunk.shape[0], -1)

            new_configs_list.append(configs_chunk)

            with torch.no_grad():
                for i in range(configs_chunk.shape[0]):
                    z_val = self.model.latent(configs_chunk[i : i + 1]).item()
                    self.opes.deposit_kernel(z_val)

            # Refresh V_OPES TorchForce with updated kernels
            try:
                self.openmm_sim.refresh_bias(
                    self.model, opes_state=self.opes.state_dict(),
                )
            except ImportError:
                pass

        new_configs = torch.cat(new_configs_list, dim=0)

        # Importance weights must match the bias used in OpenMM (TorchForce).
        if self._bias_type == "umbrella":
            target = float(self._bias_params.get("target", 0.5))
            kappa = float(self._bias_params.get("spring_constant", 100.0))
            # MACE and other large encoders OOM on wide batches for q(x).
            infer_chunk = int(self._bias_params.get("inference_max_batch", 4))
            infer_chunk = max(1, infer_chunk)
            with torch.no_grad():
                ncfg = new_configs.shape[0]
                q_parts: list[torch.Tensor] = []
                for s in range(0, ncfg, infer_chunk):
                    e = min(s + infer_chunk, ncfg)
                    q_parts.append(self.model(new_configs[s:e]))
                q = torch.cat(q_parts, dim=0)
                if q.dim() > 1:
                    q = q.squeeze(-1)
                v_eff = 0.5 * kappa * (q - target) ** 2
        elif self._bias_type == "kolmogorov":
            # V_K only (no OPES in OpenMM)
            with torch.enable_grad():
                v_eff = self.vk.bias_energy(new_configs)
            v_eff = v_eff.detach()
        elif self._bias_type == "none":
            w0 = next(self.model.parameters())
            v_eff = torch.zeros(
                new_configs.shape[0],
                device=w0.device,
                dtype=w0.dtype,
            )
        else:
            # combined_vk_opes: V_K + V_OPES (V_K uses autograd.create_graph)
            with torch.enable_grad():
                v_eff = self.combined.bias_energy(new_configs)
            v_eff = v_eff.detach()
        log_w = self._beta * v_eff

        self._config_buffer.append(new_configs.detach().cpu())
        self._log_weight_buffer.append(log_w.detach().cpu())

        # Enforce buffer cap
        if self.max_buffer_size is not None:
            self._trim_buffer()

        # Assemble full accumulated dataset
        all_configs = torch.cat(self._config_buffer, dim=0)
        all_log_w = torch.cat(self._log_weight_buffer, dim=0)

        # Normalize weights via log-sum-exp for numerical stability
        all_log_w = all_log_w - all_log_w.max()
        weights = torch.exp(all_log_w)
        weights = weights / weights.sum()

        device = next(self.model.parameters()).device
        all_configs = all_configs.to(device)
        weights = weights.to(device)

        logger.info(
            "Collected %d new snapshots (%d total accumulated), "
            "%d OPES kernels",
            new_configs.shape[0],
            all_configs.shape[0],
            self.opes.n_kernels,
        )
        return all_configs, weights

    def _trim_buffer(self) -> None:
        """Drop oldest entries if buffer exceeds max_buffer_size."""
        assert self.max_buffer_size is not None
        total = sum(c.shape[0] for c in self._config_buffer)
        while total > self.max_buffer_size and len(self._config_buffer) > 1:
            dropped = self._config_buffer.pop(0)
            self._log_weight_buffer.pop(0)
            total -= dropped.shape[0]

    def update_bias(self) -> None:
        """Re-serialize model + current OPES state and swap TorchForce."""
        try:
            self.openmm_sim.refresh_bias(
                self.model, opes_state=self.opes.state_dict(),
            )
            logger.debug("Refreshed OpenMM bias after training step")
        except ImportError:
            logger.warning("openmm-torch not available; skipping bias refresh")

    @property
    def diagnostics(self) -> dict[str, float]:
        """Return OPES convergence diagnostics."""
        total_configs = sum(c.shape[0] for c in self._config_buffer)
        return {
            "n_kernels": self.opes.n_kernels,
            "rct": self.opes.rct,
            "ess": self.opes.effective_sample_size,
            "n_accumulated": total_configs,
        }

    def opes_state_dict(self) -> dict:
        """Full checkpoint state: OPES kernels + accumulated buffer."""
        return {
            "opes": self.opes.state_dict(),
            "config_buffer": self._config_buffer,
            "log_weight_buffer": self._log_weight_buffer,
        }

    def load_opes_state_dict(self, state: dict) -> None:
        """Restore OPES kernels and accumulated buffer from checkpoint."""
        self.opes.load_state_dict(state["opes"])
        self._config_buffer = state.get("config_buffer", [])
        self._log_weight_buffer = state.get("log_weight_buffer", [])


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
