"""Tests for OpenMM simulator wrapper.

These tests verify the PyTorch modules (CommittorBiasModule, CommittorCVModule)
without requiring OpenMM to be installed.  OpenMM integration tests are
marked with @pytest.mark.openmm.
"""

import torch
import pytest

from committorch.models.mlp import CommittorMLP
from committorch.simulators.openmm_sim import (
    CommittorBiasModule,
    CommittorCVModule,
    OpenMMSimulator,
)


class TestCommittorBiasModule:
    def test_no_bias_returns_zero(self):
        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        bias_mod = CommittorBiasModule(model, bias_type="none")
        pos = torch.randn(2, 3)
        e = bias_mod(pos)
        assert e.shape == ()
        assert e.item() == pytest.approx(0.0)

    def test_umbrella_bias_nonnegative(self):
        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        bias_mod = CommittorBiasModule(
            model,
            bias_type="umbrella",
            bias_params={"target": 0.5, "spring_constant": 100.0},
        )
        pos = torch.randn(2, 3)
        e = bias_mod(pos)
        assert e.shape == ()
        assert e.item() >= 0

    def test_umbrella_bias_minimum_at_target(self):
        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        bias_mod = CommittorBiasModule(
            model,
            bias_type="umbrella",
            bias_params={"target": 0.5, "spring_constant": 100.0},
        )
        energies = []
        for _ in range(50):
            pos = torch.randn(2, 3)
            energies.append(bias_mod(pos).item())
        assert min(energies) >= 0


class TestCommittorCVModule:
    def test_cv_returns_scalar(self):
        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        cv_mod = CommittorCVModule(model)
        pos = torch.randn(2, 3)
        z = cv_mod(pos)
        assert z.shape == ()

    def test_cv_is_latent(self):
        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        cv_mod = CommittorCVModule(model)
        pos = torch.randn(2, 3)
        z_cv = cv_mod(pos)
        z_direct = model.latent(pos.reshape(1, -1))
        torch.testing.assert_close(z_cv, z_direct.squeeze())


class TestOpenMMSimulatorInterface:
    def test_no_simulation_raises_on_step(self):
        sim = OpenMMSimulator(simulation=None, kT=2.5, dim=6)
        with pytest.raises(RuntimeError):
            sim.step(torch.randn(1, 6))

    def test_no_simulation_potential_energy(self):
        sim = OpenMMSimulator(simulation=None, kT=2.5, dim=6)
        e = sim.potential_energy(torch.randn(1, 6))
        assert e.item() == 0.0

    def test_no_model_add_bias_raises(self):
        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        sim = OpenMMSimulator(simulation=None, model=model, kT=2.5, dim=6)
        with pytest.raises((ValueError, ImportError)):
            sim.add_committor_bias()
