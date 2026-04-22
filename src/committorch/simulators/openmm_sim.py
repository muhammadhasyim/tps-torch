"""OpenMM simulator wrapper with openmm-torch integration.

Wraps an OpenMM ``Simulation`` object and uses the ``openmm-torch``
plugin to apply NN-derived biases (V_K, umbrella, OPES hills) as
custom forces during simulation.

The key components:
1. ``TorchForce``: wraps a traced PyTorch model as an OpenMM force
2. ``CustomCVForce``: uses z(x) from the committor as a collective variable
3. ``Metadynamics``: OpenMM's built-in metadynamics with TorchForce-based BiasVariable

Requirements
------------
- openmm >= 8.0
- openmm-torch >= 1.5 (provides openmmtorch.TorchForce)

References
----------
.. [1] https://github.com/openmm/openmm-torch
"""

from __future__ import annotations

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
        pos = positions.reshape(1, -1)
        if self.bias_type == "umbrella":
            q = self.model(pos)
            target = self.bias_params.get("target", 0.5)
            kappa = self.bias_params.get("spring_constant", 100.0)
            return 0.5 * kappa * (q.squeeze() - target) ** 2
        elif self.bias_type == "kolmogorov":
            from ..sampling.kolmogorov import KolmogorovBias
            lam = self.bias_params.get("lambda", 1.0)
            beta = self.bias_params.get("beta", 1.0)
            vk = KolmogorovBias(self.model, lam=lam, beta=beta)
            return vk.bias_energy(pos).squeeze()
        else:
            return torch.tensor(0.0, dtype=positions.dtype)


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
        pos = positions.reshape(1, -1)
        return self.model.latent(pos).squeeze()


class OpenMMSimulator(Simulator):
    """OpenMM-based simulator with optional NN bias forces.

    Parameters
    ----------
    simulation : openmm.app.Simulation or None
        Pre-configured OpenMM simulation.  None for testing without OpenMM.
    model : CommittorModel or None
        Committor model for bias forces.
    kT : float
        Thermal energy in kJ/mol.
    """

    def __init__(
        self,
        simulation: Any = None,
        model: CommittorModel | None = None,
        kT: float = 2.494,
        dim: int = 0,
    ) -> None:
        if simulation is not None and HAS_OPENMM:
            n_atoms = simulation.topology.getNumAtoms()
            dim = n_atoms * 3
        super().__init__(dim=dim, kT=kT)
        self.simulation = simulation
        self.model = model
        self._torch_force_idx: int | None = None

    def add_committor_bias(
        self,
        bias_type: str = "umbrella",
        bias_params: dict | None = None,
    ) -> None:
        """Add a TorchForce bias to the OpenMM system.

        Parameters
        ----------
        bias_type : str
        bias_params : dict
        """
        if not HAS_OPENMM_TORCH:
            raise ImportError("openmm-torch is required for NN bias forces")
        if self.model is None:
            raise ValueError("No committor model set")
        if self.simulation is None:
            raise ValueError("No OpenMM simulation set")

        bias_module = CommittorBiasModule(self.model, bias_type, bias_params)
        traced = torch.jit.script(bias_module)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            traced.save(f.name)
            torch_force = TorchForce(f.name)

        system = self.simulation.system
        if self._torch_force_idx is not None:
            system.removeForce(self._torch_force_idx)
        self._torch_force_idx = system.addForce(torch_force)
        self.simulation.context.reinitialize(preserveState=True)

    def step(self, x: torch.Tensor, n_steps: int = 1) -> torch.Tensor:
        """Advance the OpenMM simulation by n_steps.

        Parameters
        ----------
        x : torch.Tensor
            Ignored for OpenMM (state is maintained internally).
            Provided for interface compatibility.
        n_steps : int
            Number of MD steps.

        Returns
        -------
        torch.Tensor
            Current positions as (1, n_atoms*3) tensor.
        """
        if self.simulation is None:
            raise RuntimeError("No OpenMM simulation configured")

        self.simulation.step(n_steps)
        state = self.simulation.context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)
        import numpy as np
        pos_array = np.array(positions.value_in_unit(positions.unit))
        return torch.tensor(pos_array.reshape(1, -1), dtype=torch.float32)

    def potential_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Get the current potential energy from OpenMM."""
        if self.simulation is None:
            return torch.tensor(0.0)
        state = self.simulation.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy()
        import openmm.unit as unit
        return torch.tensor(energy.value_in_unit(unit.kilojoules_per_mole))

    def force(self, x: torch.Tensor) -> torch.Tensor:
        """Get forces from OpenMM (not typically called directly)."""
        if self.simulation is None:
            return torch.zeros_like(x)
        state = self.simulation.context.getState(getForces=True)
        forces = state.getForces(asNumpy=True)
        import numpy as np
        import openmm.unit as unit
        f_array = np.array(forces.value_in_unit(unit.kilojoules_per_mole / unit.nanometer))
        return torch.tensor(f_array.reshape(x.shape), dtype=torch.float32)
