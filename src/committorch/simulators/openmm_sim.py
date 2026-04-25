"""OpenMM simulator wrapper with openmm-torch integration.

Wraps an OpenMM ``Simulation`` object and uses the ``openmm-torch``
plugin to apply NN-derived biases (V_K, umbrella, OPES) as custom
forces during simulation.

The feedback loop works as follows:
1. ``add_committor_bias()``: serialize model, create TorchForce, add to System
2. ``step()``: run MD with the current bias
3. ``get_positions()``: extract current positions as a torch tensor
4. (train model in Python)
5. ``refresh_bias()``: re-serialize updated model, swap TorchForce, reinitialize

Requirements
------------
- openmm >= 8.0
- openmm-torch >= 1.5 (provides openmmtorch.TorchForce)

References
----------
.. [1] https://github.com/openmm/openmm-torch
"""

from __future__ import annotations

from ..conda_ld_path import ensure_conda_lib_path

ensure_conda_lib_path()

import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .base import Simulator
from ..models.base import CommittorModel

HAS_OPENMM = False
try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    HAS_OPENMM = True
except ImportError:
    pass

HAS_OPENMM_TORCH = False
try:
    from openmmtorch import TorchForce
    HAS_OPENMM_TORCH = True
except ImportError:
    pass


class BiasType(str, Enum):
    """Types of bias that can be applied via TorchForce."""
    NONE = "none"
    UMBRELLA = "umbrella"
    KOLMOGOROV = "kolmogorov"
    COMBINED_VK_OPES = "combined_vk_opes"


def _kolmogorov_bias_energy(
    model: CommittorModel,
    pos: torch.Tensor,
    lam: float,
    beta: float,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Kolmogorov :math:`V_K` using ``torch.autograd.grad`` with explicit gradient scope.

    ``torch.func.jacrev`` was previously used here but its internal ``vmap`` creates
    CPU-resident accumulation buffers even when the model is on CUDA, causing a
    device mismatch when the traced module is called by TorchForce on GPU.

    ``torch.autograd.grad`` avoids ``vmap`` entirely — it computes a single VJP
    (one backward pass) which is O(n_params) instead of O(n_input) and is fully
    JIT-traceable on any device.  ``torch.enable_grad()`` overrides any enclosing
    ``torch.no_grad()`` context (e.g. during ``torch.jit.trace``), so the input
    tensor can accumulate a gradient for the backward call.

    ``create_graph=True`` is required so that the returned ``grad_z`` retains a
    ``grad_fn``; without it ``grad_z_sq`` would be a leaf tensor with no gradient
    and TorchForce would fail when it tries to differentiate the energy to get forces.

    Note
    ----
    For large systems (≫ 100 atoms) the Hessian-level computation implied by
    second-order differentiation via TorchForce is expensive.  Use
    ``bias_type="umbrella"`` for production runs on large proteins; Kolmogorov
    is correct and efficient for small test cases.
    """
    import torch.nn.functional as F

    param_dtype = next(model.parameters()).dtype
    param_device = next(model.parameters()).device
    x = pos.reshape(1, -1).to(device=param_device, dtype=param_dtype)

    with torch.enable_grad():
        x_grad = x.detach().requires_grad_(True)
        z = model.latent(x_grad)
        (grad_z,) = torch.autograd.grad(
            z.sum(), x_grad, create_graph=True
        )

    grad_z_sq = (grad_z**2).sum()
    p = model.sigmoid_steepness
    log_grad_z_sq = torch.log(grad_z_sq + eps)
    softplus_term = 4.0 * F.softplus(-p * z)
    linear_term = 2.0 * p * z
    return (lam / beta) * (log_grad_z_sq - softplus_term - linear_term).squeeze()


class CommittorBiasModule(nn.Module):
    """PyTorch module that computes a bias potential from positions.

    Wraps a committor model's bias computation so it can be traced
    and used with openmm-torch's TorchForce.

    Parameters
    ----------
    model : CommittorModel
        Committor network.
    bias_type : str
        Type of bias: "umbrella", "kolmogorov", or "none".
    bias_params : dict
        Parameters for the bias (e.g., target, spring_constant for umbrella).
    """

    def __init__(
        self,
        model: CommittorModel,
        bias_type: str = "none",
        bias_params: dict | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.bias_type = bias_type
        self.bias_params = bias_params or {}

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute bias energy from positions.

        Parameters
        ----------
        positions : torch.Tensor, shape (n_atoms, 3) or (n_atoms*3,)
            Atomic positions (single configuration, no batch dim).

        Returns
        -------
        torch.Tensor
            Scalar bias energy.
        """
        param_dtype = next(self.model.parameters()).dtype
        param_device = next(self.model.parameters()).device
        # Cast both dtype and device: OpenMM CUDA passes float64 CUDA tensors;
        # the model may be on a different device than the raw positions.
        positions = positions.to(device=param_device, dtype=param_dtype)
        pos = positions.reshape(1, -1)
        if self.bias_type == "umbrella":
            q = self.model(pos)
            target = self.bias_params.get("target", 0.5)
            kappa = self.bias_params.get("spring_constant", 100.0)
            return 0.5 * kappa * (q.squeeze() - target) ** 2
        elif self.bias_type == "kolmogorov":
            lam = self.bias_params.get("lambda", 1.0)
            beta = self.bias_params.get("beta", 1.0)
            eps = self.bias_params.get("eps", 1e-10)
            return _kolmogorov_bias_energy(
                self.model, pos, lam=lam, beta=beta, eps=eps
            )
        else:
            return torch.tensor(0.0, dtype=param_dtype, device=param_device)


class OPESBiasModule(nn.Module):
    r"""OPES bias V_OPES(z(x)) for TorchForce (``torch.jit.trace``-safe).

    Encodes the current OPES kernel state as registered buffers so the
    KDE evaluation is pure tensor math with no Python loops.
    Re-trace (via ``refresh_bias``) whenever the kernel set changes.

    This module computes **only** V_OPES.  For the combined V_K + V_OPES
    scheme, OpenMM uses *two* TorchForce objects: one for V_K
    (``CommittorBiasModule`` with ``bias_type="kolmogorov"``) and one
    for V_OPES (this class).  OpenMM natively sums forces from all Force
    objects, giving the correct V_eff = V_K + V_OPES without needing to
    put second-order-gradient V_K in the same traced graph as V_OPES.

    Parameters
    ----------
    model : CommittorModel
        Committor network providing z(x).
    beta : float
        Inverse temperature 1/(k_B T).
    opes_state : dict
        Snapshot from ``OPESBias.state_dict()``.
    """

    def __init__(
        self,
        model: CommittorModel,
        beta: float = 1.0,
        opes_state: dict | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.beta = beta

        if opes_state is None:
            opes_state = {
                "kernel_centers": torch.tensor([]),
                "kernel_sigmas": torch.tensor([]),
                "kernel_weights": torch.tensor([]),
                "sum_weights": 1.0,
                "Zed": 1.0,
                "bias_prefactor": 0.9,
                "epsilon": 1e-6,
            }

        self.register_buffer(
            "opes_centers", opes_state["kernel_centers"].float()
        )
        self.register_buffer(
            "opes_sigmas", opes_state["kernel_sigmas"].float()
        )
        self.register_buffer(
            "opes_weights", opes_state["kernel_weights"].float()
        )
        self.opes_sum_weights = float(opes_state["sum_weights"])
        self.opes_Zed = float(opes_state["Zed"])
        self.opes_bias_prefactor = float(
            opes_state.get("bias_prefactor", 0.9)
        )
        self.opes_epsilon = float(opes_state.get("epsilon", 1e-6))

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute V_OPES(z(x)) as a scalar energy."""
        param_dtype = next(self.model.parameters()).dtype
        param_device = next(self.model.parameters()).device
        positions = positions.to(device=param_device, dtype=param_dtype)
        z = self.model.latent(positions.reshape(1, -1)).squeeze()

        dz = z - self.opes_centers
        gaussians = torch.exp(-0.5 * (dz / (self.opes_sigmas + 1e-30)) ** 2)
        norm = 1.0 / (
            self.opes_sigmas * 2.5066282746310002 + 1e-30
        )
        prob = (self.opes_weights * norm * gaussians).sum()
        prob = prob / (self.opes_sum_weights + 1e-30)

        kT = 1.0 / self.beta
        return kT * self.opes_bias_prefactor * torch.log(
            prob / (self.opes_Zed + 1e-30) + self.opes_epsilon
        )


class CommittorCVModule(nn.Module):
    """PyTorch module exposing z(x) as a collective variable for OpenMM.

    Parameters
    ----------
    model : CommittorModel
        Committor network.
    """

    def __init__(self, model: CommittorModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Return z(x), the pre-sigmoid latent, as a scalar CV."""
        param_dtype = next(self.model.parameters()).dtype
        param_device = next(self.model.parameters()).device
        positions = positions.to(device=param_device, dtype=param_dtype)
        pos = positions.reshape(1, -1)
        return self.model.latent(pos).squeeze()


def _trace_staging_dir() -> str | None:
    """Writable directory for large TorchScript traces.

    Default system temp (often ``/tmp`` on the root filesystem) may be
    full on HPC nodes while the project disk still has space. Prefer
    ``COMMITTORCH_JIT_TMPDIR``, then the process current working directory.
    """
    env = os.environ.get("COMMITTORCH_JIT_TMPDIR")
    if env:
        p = Path(env)
        try:
            p.mkdir(parents=True, exist_ok=True)
            if os.access(p, os.W_OK):
                return str(p)
        except OSError:
            pass
    cwd = Path.cwd()
    if cwd.is_dir() and os.access(cwd, os.W_OK):
        return str(cwd)
    return None


def _serialize_bias_module(
    model: CommittorModel,
    bias_type: str = "none",
    bias_params: dict | None = None,
    n_atoms: int = 2,
) -> str:
    """Trace a CommittorBiasModule and save to a temp file.

    Uses ``torch.jit.trace`` rather than ``torch.jit.script`` to
    avoid JIT limitations (inline imports, dynamic dispatch).

    Parameters
    ----------
    n_atoms : int
        Number of atoms, used to construct the example input for tracing.

    Returns
    -------
    str
        Path to the saved .pt file.
    """
    bias_module = CommittorBiasModule(model, bias_type, bias_params)
    bias_module.eval()
    # Trace on the model's own device so JIT ops are baked in for the correct
    # device (CUDA or CPU).  TorchForce will call this module with positions on
    # the same device as the OpenMM platform (CUDA when using Platform=CUDA).
    param_device = next(model.parameters()).device
    example_input = torch.randn(n_atoms, 3, device=param_device)
    # Kolmogorov bias uses torch.autograd.grad with create_graph=True;
    # wrapping in no_grad breaks tracing for that type.
    if bias_type == "kolmogorov":
        traced = torch.jit.trace(bias_module, example_input)
    else:
        with torch.no_grad():
            traced = torch.jit.trace(bias_module, example_input)
    staging = _trace_staging_dir()
    tmp = tempfile.NamedTemporaryFile(
        suffix=".pt", delete=False, dir=staging if staging else None
    )
    traced.save(tmp.name)
    tmp.close()
    return tmp.name


def _serialize_opes_bias(
    model: CommittorModel,
    opes_state: dict,
    beta: float = 1.0,
    n_atoms: int = 2,
) -> str:
    """Trace an OPESBiasModule and save to a temp file.

    Parameters
    ----------
    model : CommittorModel
    opes_state : dict
        From ``OPESBias.state_dict()``.
    beta : float
        Inverse temperature.
    n_atoms : int
        Number of atoms for the example trace input.

    Returns
    -------
    str
        Path to the saved .pt file.
    """
    module = OPESBiasModule(
        model=model,
        beta=beta,
        opes_state=opes_state,
    )
    module.eval()
    param_device = next(model.parameters()).device
    example_input = torch.randn(n_atoms, 3, device=param_device)
    with torch.no_grad():
        traced = torch.jit.trace(module, example_input)
    staging = _trace_staging_dir()
    tmp = tempfile.NamedTemporaryFile(
        suffix=".pt", delete=False, dir=staging if staging else None,
    )
    traced.save(tmp.name)
    tmp.close()
    return tmp.name


class OpenMMSimulator(Simulator):
    """OpenMM-based simulator with NN bias forces via openmm-torch.

    Supports the feedback loop: the committor model can be re-injected
    after training via ``refresh_bias()``, which re-serializes the model
    and swaps the TorchForce without losing the simulation state.

    Parameters
    ----------
    simulation : openmm.app.Simulation or None
        Pre-configured OpenMM simulation. None for testing without OpenMM.
    model : CommittorModel or None
        Committor model for bias forces.
    kT : float
        Thermal energy in kJ/mol (default: 2.494 ~ 300 K).
    dim : int
        Dimensionality override (auto-detected from simulation if available).
    flatten_positions : bool
        If True, ``step()`` and ``get_positions()`` return (1, n_atoms*3).
        If False, they return (1, n_atoms, 3).
    """

    def __init__(
        self,
        simulation: Any = None,
        model: CommittorModel | None = None,
        kT: float = 2.494,
        dim: int = 0,
        flatten_positions: bool = True,
    ) -> None:
        self._n_atoms: int = 0
        if simulation is not None and HAS_OPENMM:
            self._n_atoms = simulation.topology.getNumAtoms()
            dim = self._n_atoms * 3
        super().__init__(dim=dim, kT=kT)
        self.simulation = simulation
        self.model = model
        self.flatten_positions = flatten_positions
        self._torch_force_idx: int | None = None
        self._opes_force_idx: int | None = None
        self._bias_type: str = "none"
        self._bias_params: dict = {}
        self._opes_state: dict | None = None
        self._serialized_model_path: str | None = None
        self._serialized_opes_path: str | None = None

    @property
    def n_atoms(self) -> int:
        return self._n_atoms

    def get_positions(self) -> torch.Tensor:
        """Extract current positions from the OpenMM context.

        Returns
        -------
        torch.Tensor
            Shape (1, n_atoms*3) if flatten_positions, else (1, n_atoms, 3).
        """
        if self.simulation is None:
            raise RuntimeError("No OpenMM simulation configured")

        state = self.simulation.context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)
        import numpy as np
        pos_nm = np.array(positions.value_in_unit(positions.unit))
        pos_t = torch.tensor(pos_nm, dtype=torch.float32)
        if self.model is not None:
            param_device = next(self.model.parameters()).device
            pos_t = pos_t.to(device=param_device)
        if self.flatten_positions:
            return pos_t.reshape(1, -1)
        return pos_t.unsqueeze(0)

    def set_positions(self, positions: torch.Tensor) -> None:
        """Set positions in the OpenMM context.

        Parameters
        ----------
        positions : torch.Tensor
            Shape (n_atoms, 3), (1, n_atoms, 3), or (1, n_atoms*3).
        """
        if self.simulation is None:
            raise RuntimeError("No OpenMM simulation configured")

        pos = positions.detach().cpu()
        if pos.dim() == 3:
            pos = pos.squeeze(0)
        elif pos.dim() == 2 and pos.shape[0] == 1:
            pos = pos.reshape(self._n_atoms, 3)

        pos_list = pos.numpy().tolist()
        self.simulation.context.setPositions(pos_list)

    def add_committor_bias(
        self,
        bias_type: str = "umbrella",
        bias_params: dict | None = None,
        opes_state: dict | None = None,
    ) -> None:
        """Add a TorchForce bias to the OpenMM system.

        For ``combined_vk_opes``, two TorchForce objects are added:
        one for V_K (kolmogorov, second-order grad) and one for V_OPES
        (first-order, KDE-based).  OpenMM sums forces from both.

        Parameters
        ----------
        bias_type : str
            One of "umbrella", "kolmogorov", "combined_vk_opes", "none".
        bias_params : dict
            Bias-specific parameters.
        opes_state : dict or None
            OPES kernel state from ``OPESBias.state_dict()``.
            Required when ``bias_type="combined_vk_opes"``.
        """
        if not HAS_OPENMM_TORCH:
            raise ImportError("openmm-torch is required for NN bias forces")
        if self.model is None:
            raise ValueError("No committor model set")
        if self.simulation is None:
            raise ValueError("No OpenMM simulation set")

        self._bias_type = bias_type
        self._bias_params = bias_params or {}
        self._opes_state = opes_state

        n_atoms = max(self._n_atoms, 2)
        system = self.simulation.system

        # Remove any existing bias forces
        for idx_attr in ("_torch_force_idx", "_opes_force_idx"):
            idx = getattr(self, idx_attr)
            if idx is not None:
                system.removeForce(idx)
                setattr(self, idx_attr, None)
        # Force indices shift after removal; reinitialize to be safe
        if self._torch_force_idx is not None or self._opes_force_idx is not None:
            self.simulation.context.reinitialize(preserveState=True)
            self._torch_force_idx = None
            self._opes_force_idx = None

        if bias_type == "combined_vk_opes":
            # Two TorchForces: V_K (kolmogorov) + V_OPES (OPES-only)
            vk_path = _serialize_bias_module(
                self.model, "kolmogorov", self._bias_params, n_atoms=n_atoms,
            )
            self._serialized_model_path = vk_path
            vk_force = TorchForce(vk_path)
            self._torch_force_idx = system.addForce(vk_force)

            beta = self._bias_params.get("beta", 1.0)
            opes_st = opes_state if opes_state is not None else {
                "kernel_centers": torch.tensor([]),
                "kernel_sigmas": torch.tensor([]),
                "kernel_weights": torch.tensor([]),
                "sum_weights": 1.0, "Zed": 1.0,
                "bias_prefactor": 0.9, "epsilon": 1e-6,
            }
            opes_path = _serialize_opes_bias(
                self.model, opes_st, beta=beta, n_atoms=n_atoms,
            )
            self._serialized_opes_path = opes_path
            opes_force = TorchForce(opes_path)
            self._opes_force_idx = system.addForce(opes_force)
        else:
            pt_path = _serialize_bias_module(
                self.model, bias_type, self._bias_params, n_atoms=n_atoms,
            )
            self._serialized_model_path = pt_path
            torch_force = TorchForce(pt_path)
            self._torch_force_idx = system.addForce(torch_force)

        self.simulation.context.reinitialize(preserveState=True)

    def refresh_bias(
        self,
        model: CommittorModel | None = None,
        opes_state: dict | None = None,
    ) -> None:
        """Re-serialize the committor model and swap the TorchForce(s).

        Call this after training to inject the updated model into the
        running OpenMM simulation. Preserves positions and velocities.

        For ``combined_vk_opes``, both the V_K and V_OPES TorchForce
        modules are re-serialized and swapped.

        Parameters
        ----------
        model : CommittorModel or None
            Updated model. If None, uses ``self.model``.
        opes_state : dict or None
            Updated OPES kernel state. If None, uses the previously
            stored ``self._opes_state`` (for combined_vk_opes).
        """
        if model is not None:
            self.model = model
        if opes_state is not None:
            self._opes_state = opes_state

        if self.model is None:
            raise ValueError("No committor model to refresh")
        if self.simulation is None:
            raise RuntimeError("No OpenMM simulation configured")
        if not HAS_OPENMM_TORCH:
            raise ImportError("openmm-torch is required for NN bias forces")

        n_atoms = max(self._n_atoms, 2)
        system = self.simulation.system
        old_paths: list[str | None] = []

        if self._bias_type == "combined_vk_opes":
            # Re-serialize V_K
            vk_path = _serialize_bias_module(
                self.model, "kolmogorov", self._bias_params, n_atoms=n_atoms,
            )
            old_paths.append(self._serialized_model_path)
            self._serialized_model_path = vk_path

            if self._torch_force_idx is not None:
                system.removeForce(self._torch_force_idx)
            self._torch_force_idx = system.addForce(TorchForce(vk_path))

            # Re-serialize V_OPES
            beta = self._bias_params.get("beta", 1.0)
            cur_opes = self._opes_state or {
                "kernel_centers": torch.tensor([]),
                "kernel_sigmas": torch.tensor([]),
                "kernel_weights": torch.tensor([]),
                "sum_weights": 1.0, "Zed": 1.0,
                "bias_prefactor": 0.9, "epsilon": 1e-6,
            }
            opes_path = _serialize_opes_bias(
                self.model, cur_opes, beta=beta, n_atoms=n_atoms,
            )
            old_paths.append(self._serialized_opes_path)
            self._serialized_opes_path = opes_path

            if self._opes_force_idx is not None:
                system.removeForce(self._opes_force_idx)
            self._opes_force_idx = system.addForce(TorchForce(opes_path))
        else:
            pt_path = _serialize_bias_module(
                self.model, self._bias_type, self._bias_params,
                n_atoms=n_atoms,
            )
            old_paths.append(self._serialized_model_path)
            self._serialized_model_path = pt_path

            if self._torch_force_idx is not None:
                system.removeForce(self._torch_force_idx)
            self._torch_force_idx = system.addForce(TorchForce(pt_path))

        self.simulation.context.reinitialize(preserveState=True)

        for p in old_paths:
            if p is not None:
                Path(p).unlink(missing_ok=True)

    def step(self, x: torch.Tensor, n_steps: int = 1) -> torch.Tensor:
        """Advance the OpenMM simulation by n_steps.

        Parameters
        ----------
        x : torch.Tensor
            Ignored for OpenMM (state is maintained internally).
            Provided for interface compatibility with ``Simulator``.
        n_steps : int
            Number of MD steps.

        Returns
        -------
        torch.Tensor
            Current positions, shape depends on ``flatten_positions``.
        """
        if self.simulation is None:
            raise RuntimeError("No OpenMM simulation configured")

        self.simulation.step(n_steps)
        return self.get_positions()

    def step_and_collect(
        self,
        n_steps: int,
        collect_interval: int = 1,
    ) -> torch.Tensor:
        """Run MD and collect snapshots at regular intervals.

        Parameters
        ----------
        n_steps : int
            Total MD steps to run.
        collect_interval : int
            Collect a snapshot every this many steps.

        Returns
        -------
        torch.Tensor
            Collected positions, shape (n_snapshots, n_atoms*3) or
            (n_snapshots, n_atoms, 3) depending on ``flatten_positions``.
        """
        if self.simulation is None:
            raise RuntimeError("No OpenMM simulation configured")

        snapshots: list[torch.Tensor] = []
        steps_done = 0
        while steps_done < n_steps:
            chunk = min(collect_interval, n_steps - steps_done)
            self.simulation.step(chunk)
            steps_done += chunk
            snapshots.append(self.get_positions())

        return torch.cat(snapshots, dim=0)

    def potential_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Get the current potential energy from OpenMM."""
        if self.simulation is None:
            return torch.tensor(0.0, dtype=torch.float32)
        state = self.simulation.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy()
        import openmm.unit as u
        return torch.tensor(
            energy.value_in_unit(u.kilojoules_per_mole), dtype=torch.float32
        )

    def force(self, x: torch.Tensor) -> torch.Tensor:
        """Get forces from OpenMM."""
        if self.simulation is None:
            return torch.zeros_like(x)
        state = self.simulation.context.getState(getForces=True)
        forces = state.getForces(asNumpy=True)
        import numpy as np
        import openmm.unit as u
        f_array = np.array(
            forces.value_in_unit(u.kilojoules_per_mole / u.nanometer)
        )
        return torch.tensor(f_array.reshape(x.shape), dtype=torch.float32)
