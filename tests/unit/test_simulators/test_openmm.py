"""Tests for OpenMM simulator wrapper.

These tests verify the PyTorch modules (CommittorBiasModule, CommittorCVModule)
and OpenMMSimulator interface without requiring OpenMM to be installed.
OpenMM integration tests would be marked with @pytest.mark.openmm.
"""

import torch
import pytest
from unittest.mock import MagicMock, patch

from committorch.models.mlp import CommittorMLP
from committorch.simulators.openmm_sim import (
    BiasType,
    CommittorBiasModule,
    CommittorCVModule,
    OpenMMSimulator,
    _serialize_bias_module,
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

    def test_flat_input_shape(self):
        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        bias_mod = CommittorBiasModule(model, bias_type="umbrella",
                                       bias_params={"target": 0.3})
        pos = torch.randn(6)
        e = bias_mod(pos)
        assert e.shape == ()


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


class TestBiasType:
    def test_enum_values(self):
        assert BiasType.NONE == "none"
        assert BiasType.UMBRELLA == "umbrella"
        assert BiasType.KOLMOGOROV == "kolmogorov"
        assert BiasType.COMBINED_VK_OPES == "combined_vk_opes"


class TestSerializeBiasModule:
    def test_serialize_creates_file(self, tmp_path):
        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        path = _serialize_bias_module(model, bias_type="none")
        from pathlib import Path
        assert Path(path).exists()
        Path(path).unlink()

    def test_serialize_umbrella(self, tmp_path):
        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        path = _serialize_bias_module(
            model, bias_type="umbrella",
            bias_params={"target": 0.5, "spring_constant": 50.0}
        )
        from pathlib import Path
        assert Path(path).exists()
        Path(path).unlink()

    def test_serialize_kolmogorov(self, tmp_path):
        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        path = _serialize_bias_module(
            model,
            bias_type="kolmogorov",
            bias_params={"lambda": 1.0, "beta": 2.5, "eps": 1e-10},
        )
        from pathlib import Path
        assert Path(path).exists()
        Path(path).unlink()

    def test_kolmogorov_bias_matches_kolmogorov_bias_class(self):
        from committorch.sampling.kolmogorov import KolmogorovBias

        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        bias_mod = CommittorBiasModule(
            model,
            bias_type="kolmogorov",
            bias_params={"lambda": 1.0, "beta": 0.4, "eps": 1e-10},
        )
        pos = torch.randn(2, 3, requires_grad=True)
        e_mod = bias_mod(pos)
        vk = KolmogorovBias(model, lam=1.0, beta=0.4, eps=1e-10)
        e_class = vk.bias_energy(pos.reshape(1, -1)).squeeze()
        torch.testing.assert_close(e_mod, e_class)


class TestDtypeSafety:
    """Verify that float64 inputs (as OpenMM provides) are handled correctly."""

    def test_bias_module_umbrella_float64_input(self):
        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        bias_mod = CommittorBiasModule(
            model, bias_type="umbrella",
            bias_params={"target": 0.5, "spring_constant": 100.0},
        )
        pos = torch.randn(2, 3, dtype=torch.float64)
        e = bias_mod(pos)
        assert e.shape == ()
        assert e.dtype == torch.float32

    def test_bias_module_kolmogorov_float64_input(self):
        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        bias_mod = CommittorBiasModule(
            model, bias_type="kolmogorov",
            bias_params={"lambda": 1.0, "beta": 0.4, "eps": 1e-10},
        )
        pos = torch.randn(2, 3, dtype=torch.float64)
        e = bias_mod(pos)
        assert e.shape == ()
        assert e.dtype == torch.float32

    def test_bias_module_none_float64_input(self):
        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        bias_mod = CommittorBiasModule(model, bias_type="none")
        pos = torch.randn(2, 3, dtype=torch.float64)
        e = bias_mod(pos)
        assert e.item() == pytest.approx(0.0)
        assert e.dtype == torch.float32

    def test_cv_module_float64_input(self):
        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        cv_mod = CommittorCVModule(model)
        pos_f32 = torch.randn(2, 3)
        pos_f64 = pos_f32.double()
        z_f32 = cv_mod(pos_f32)
        z_f64 = cv_mod(pos_f64)
        torch.testing.assert_close(z_f32, z_f64, atol=1e-5, rtol=1e-5)

    def test_traced_module_accepts_float64(self):
        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        path = _serialize_bias_module(
            model, bias_type="umbrella",
            bias_params={"target": 0.5, "spring_constant": 50.0},
        )
        from pathlib import Path
        traced = torch.jit.load(path)
        pos_f64 = torch.randn(2, 3, dtype=torch.float64)
        e = traced(pos_f64)
        assert e.shape == ()
        assert torch.isfinite(e)
        Path(path).unlink()

    def test_traced_kolmogorov_accepts_float64(self):
        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        path = _serialize_bias_module(
            model, bias_type="kolmogorov",
            bias_params={"lambda": 1.0, "beta": 2.5, "eps": 1e-10},
        )
        from pathlib import Path
        traced = torch.jit.load(path)
        pos_f64 = torch.randn(2, 3, dtype=torch.float64)
        e = traced(pos_f64)
        assert e.shape == ()
        assert torch.isfinite(e)
        Path(path).unlink()

    def test_kolmogorov_float64_matches_float32(self):
        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        bias_mod = CommittorBiasModule(
            model, bias_type="kolmogorov",
            bias_params={"lambda": 1.0, "beta": 0.4, "eps": 1e-10},
        )
        pos_f32 = torch.randn(2, 3)
        pos_f64 = pos_f32.double()
        e_f32 = bias_mod(pos_f32)
        e_f64 = bias_mod(pos_f64)
        torch.testing.assert_close(e_f32, e_f64, atol=1e-4, rtol=1e-4)


class TestExportKolmogorovMatch:
    """Verify CommittorBiasForOpenMM Kolmogorov matches KolmogorovBias."""

    def test_export_kolmogorov_matches_reference(self):
        from committorch.models.export import CommittorBiasForOpenMM
        from committorch.sampling.kolmogorov import KolmogorovBias

        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        export_mod = CommittorBiasForOpenMM(
            model, bias_type="kolmogorov",
            bias_params={"lambda": 1.0, "beta": 0.4, "eps": 1e-10},
            n_atoms=2,
        )
        pos = torch.randn(2, 3, requires_grad=True)
        e_export = export_mod(pos)
        vk = KolmogorovBias(model, lam=1.0, beta=0.4, eps=1e-10)
        e_ref = vk.bias_energy(pos.reshape(1, -1)).squeeze()
        torch.testing.assert_close(e_export, e_ref)

    def test_export_kolmogorov_float64_input(self):
        from committorch.models.export import CommittorBiasForOpenMM

        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        export_mod = CommittorBiasForOpenMM(
            model, bias_type="kolmogorov",
            bias_params={"lambda": 1.0, "beta": 0.4, "eps": 1e-10},
            n_atoms=2,
        )
        pos = torch.randn(2, 3, dtype=torch.float64)
        e = export_mod(pos)
        assert e.shape == ()
        assert e.dtype == torch.float32

    def test_export_traces_and_runs(self, tmp_path):
        from committorch.models.export import CommittorBiasForOpenMM

        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        export_mod = CommittorBiasForOpenMM(
            model, bias_type="kolmogorov",
            bias_params={"lambda": 1.0, "beta": 2.5, "eps": 1e-10},
            n_atoms=2,
        )
        pt_path = export_mod.export(tmp_path / "bias.pt")
        traced = torch.jit.load(str(pt_path))
        pos = torch.randn(2, 3, dtype=torch.float64)
        e = traced(pos)
        assert e.shape == ()
        assert torch.isfinite(e)


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

    def test_no_simulation_get_positions_raises(self):
        sim = OpenMMSimulator(simulation=None, kT=2.5, dim=6)
        with pytest.raises(RuntimeError):
            sim.get_positions()

    def test_no_simulation_set_positions_raises(self):
        sim = OpenMMSimulator(simulation=None, kT=2.5, dim=6)
        with pytest.raises(RuntimeError):
            sim.set_positions(torch.randn(2, 3))

    def test_no_simulation_step_and_collect_raises(self):
        sim = OpenMMSimulator(simulation=None, kT=2.5, dim=6)
        with pytest.raises(RuntimeError):
            sim.step_and_collect(n_steps=10)

    def test_refresh_bias_no_model_raises(self):
        sim = OpenMMSimulator(simulation=None, kT=2.5, dim=6)
        with pytest.raises(ValueError):
            sim.refresh_bias()

    def test_flatten_positions_default(self):
        sim = OpenMMSimulator(simulation=None, kT=2.5, dim=6)
        assert sim.flatten_positions is True

    def test_n_atoms_no_simulation(self):
        sim = OpenMMSimulator(simulation=None, kT=2.5, dim=9)
        assert sim.n_atoms == 0

    def test_bias_type_stored(self):
        sim = OpenMMSimulator(simulation=None, kT=2.5, dim=6)
        assert sim._bias_type == "none"


class TestOpenMMSimulatorMocked:
    """Tests using a mock OpenMM Simulation to verify the feedback loop logic."""

    def _make_mock_simulation(self, n_atoms: int = 3):
        """Create a mock simulation that returns predictable positions."""
        import numpy as np

        mock_sim = MagicMock()
        mock_topology = MagicMock()
        mock_topology.getNumAtoms.return_value = n_atoms
        mock_sim.topology = mock_topology

        mock_system = MagicMock()
        mock_system.addForce.return_value = 0
        mock_sim.system = mock_system

        mock_context = MagicMock()
        mock_state = MagicMock()

        positions_array = np.random.randn(n_atoms, 3).astype(np.float64)
        mock_quantity = MagicMock()
        mock_quantity.value_in_unit.return_value = positions_array
        mock_quantity.unit = "nanometers"
        mock_state.getPositions.return_value = mock_quantity

        mock_context.getState.return_value = mock_state
        mock_sim.context = mock_context

        return mock_sim, positions_array

    @patch("committorch.simulators.openmm_sim.HAS_OPENMM", True)
    def test_step_returns_flat_positions(self):
        mock_sim, pos_array = self._make_mock_simulation(n_atoms=3)
        sim = OpenMMSimulator.__new__(OpenMMSimulator)
        sim._n_atoms = 3
        sim.dim = 9
        sim.kT = 2.5
        sim.simulation = mock_sim
        sim.model = None
        sim.flatten_positions = True
        sim._bias_fn = None
        sim._torch_force_idx = None
        sim._bias_type = "none"
        sim._bias_params = {}
        sim._serialized_model_path = None

        result = sim.step(torch.randn(1, 9), n_steps=5)
        mock_sim.step.assert_called_once_with(5)
        assert result.shape == (1, 9)

    @patch("committorch.simulators.openmm_sim.HAS_OPENMM", True)
    def test_step_returns_3d_positions(self):
        mock_sim, pos_array = self._make_mock_simulation(n_atoms=3)
        sim = OpenMMSimulator.__new__(OpenMMSimulator)
        sim._n_atoms = 3
        sim.dim = 9
        sim.kT = 2.5
        sim.simulation = mock_sim
        sim.model = None
        sim.flatten_positions = False
        sim._bias_fn = None
        sim._torch_force_idx = None
        sim._bias_type = "none"
        sim._bias_params = {}
        sim._serialized_model_path = None

        result = sim.step(torch.randn(1, 9))
        assert result.shape == (1, 3, 3)

    @patch("committorch.simulators.openmm_sim.HAS_OPENMM", True)
    def test_step_and_collect_shape(self):
        mock_sim, _ = self._make_mock_simulation(n_atoms=3)
        sim = OpenMMSimulator.__new__(OpenMMSimulator)
        sim._n_atoms = 3
        sim.dim = 9
        sim.kT = 2.5
        sim.simulation = mock_sim
        sim.model = None
        sim.flatten_positions = True
        sim._bias_fn = None
        sim._torch_force_idx = None
        sim._bias_type = "none"
        sim._bias_params = {}
        sim._serialized_model_path = None

        result = sim.step_and_collect(n_steps=10, collect_interval=5)
        assert result.shape[0] == 2
        assert result.shape[1] == 9

    @patch("committorch.simulators.openmm_sim.HAS_OPENMM", True)
    def test_get_set_positions_roundtrip(self):
        mock_sim, pos_array = self._make_mock_simulation(n_atoms=3)
        sim = OpenMMSimulator.__new__(OpenMMSimulator)
        sim._n_atoms = 3
        sim.dim = 9
        sim.kT = 2.5
        sim.simulation = mock_sim
        sim.model = None
        sim.flatten_positions = True
        sim._bias_fn = None
        sim._torch_force_idx = None
        sim._bias_type = "none"
        sim._bias_params = {}
        sim._serialized_model_path = None

        pos = sim.get_positions()
        assert pos.shape == (1, 9)

        sim.set_positions(pos)
        mock_sim.context.setPositions.assert_called_once()
