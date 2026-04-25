"""Tests for protein training loops (ProteinOPESLoop, ProteinUmbrellaLoop).

These tests use mock OpenMM simulations and small MLP models to verify
the training loop logic without requiring OpenMM or openmm-torch.
"""

import torch
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from committorch.models.mlp import CommittorMLP
from committorch.training.protein_loop import (
    ProteinOPESLoop,
    ProteinUmbrellaLoop,
)


def _make_mock_openmm_simulator(n_atoms: int = 5, kT: float = 2.5):
    """Create a mock OpenMMSimulator for testing."""
    from committorch.simulators.openmm_sim import OpenMMSimulator

    sim = OpenMMSimulator.__new__(OpenMMSimulator)
    sim._n_atoms = n_atoms
    sim.dim = n_atoms * 3
    sim.kT = kT
    sim.simulation = MagicMock()
    sim.model = None
    sim.flatten_positions = True
    sim._bias_fn = None
    sim._torch_force_idx = None
    sim._opes_force_idx = None
    sim._bias_type = "none"
    sim._bias_params = {}
    sim._opes_state = None
    sim._serialized_model_path = None
    sim._serialized_opes_path = None

    mock_topology = MagicMock()
    mock_topology.getNumAtoms.return_value = n_atoms
    sim.simulation.topology = mock_topology

    mock_system = MagicMock()
    mock_system.addForce.return_value = 0
    sim.simulation.system = mock_system

    mock_context = MagicMock()
    mock_state = MagicMock()
    positions_array = np.random.randn(n_atoms, 3).astype(np.float64)
    mock_quantity = MagicMock()
    mock_quantity.value_in_unit.return_value = positions_array
    mock_quantity.unit = "nanometers"
    mock_state.getPositions.return_value = mock_quantity
    mock_context.getState.return_value = mock_state
    sim.simulation.context = mock_context

    return sim


class TestProteinOPESLoop:
    def test_construction(self):
        n_atoms = 5
        model = CommittorMLP(input_dim=n_atoms * 3, hidden_dims=[16])
        sim = _make_mock_openmm_simulator(n_atoms=n_atoms)

        x_a = torch.randn(3, n_atoms * 3)
        x_b = torch.randn(3, n_atoms * 3)

        loop = ProteinOPESLoop(
            model=model,
            simulator=sim,
            x_a=x_a,
            x_b=x_b,
            n_sim_steps=10,
            collect_interval=5,
            kernel_deposit_interval=5,
            bias_type="umbrella",
        )

        assert loop.model is model
        assert loop.openmm_sim is sim
        assert loop.kernel_deposit_interval == 5
        assert loop.opes.n_kernels == 0

    def test_collect_samples_returns_weights(self):
        """collect_samples must return non-None importance weights."""
        n_atoms = 3
        model = CommittorMLP(input_dim=n_atoms * 3, hidden_dims=[16])
        sim = _make_mock_openmm_simulator(n_atoms=n_atoms)

        x_a = torch.randn(2, n_atoms * 3)
        x_b = torch.randn(2, n_atoms * 3)

        loop = ProteinOPESLoop(
            model=model,
            simulator=sim,
            x_a=x_a,
            x_b=x_b,
            n_sim_steps=10,
            collect_interval=5,
            kernel_deposit_interval=5,
            bias_type="umbrella",
        )

        configs, weights = loop.collect_samples()
        assert configs.dim() == 2
        assert configs.shape[1] == n_atoms * 3
        assert weights is not None, "Weights must not be None (reweighting required)"
        assert weights.shape[0] == configs.shape[0]
        assert abs(weights.sum().item() - 1.0) < 1e-4, "Weights should sum to 1"

    def test_opes_kernels_deposited(self):
        n_atoms = 3
        model = CommittorMLP(input_dim=n_atoms * 3, hidden_dims=[16])
        sim = _make_mock_openmm_simulator(n_atoms=n_atoms)

        x_a = torch.randn(2, n_atoms * 3)
        x_b = torch.randn(2, n_atoms * 3)

        loop = ProteinOPESLoop(
            model=model,
            simulator=sim,
            x_a=x_a,
            x_b=x_b,
            n_sim_steps=10,
            collect_interval=5,
            kernel_deposit_interval=1,
            bias_type="umbrella",
        )
        configs, _ = loop.collect_samples()
        assert loop.opes.n_kernels > 0

    def test_buffer_accumulates_across_iterations(self):
        """Dataset grows across collect_samples calls."""
        n_atoms = 3
        model = CommittorMLP(input_dim=n_atoms * 3, hidden_dims=[16])
        sim = _make_mock_openmm_simulator(n_atoms=n_atoms)

        loop = ProteinOPESLoop(
            model=model,
            simulator=sim,
            x_a=torch.randn(2, n_atoms * 3),
            x_b=torch.randn(2, n_atoms * 3),
            n_sim_steps=10,
            collect_interval=5,
            kernel_deposit_interval=5,
            bias_type="umbrella",
        )

        c1, w1 = loop.collect_samples()
        n1 = c1.shape[0]
        c2, w2 = loop.collect_samples()
        n2 = c2.shape[0]
        assert n2 > n1, "Buffer should grow across iterations"
        assert w2.shape[0] == n2

    def test_max_buffer_size_trims(self):
        """max_buffer_size caps the accumulated dataset."""
        n_atoms = 3
        model = CommittorMLP(input_dim=n_atoms * 3, hidden_dims=[16])
        sim = _make_mock_openmm_simulator(n_atoms=n_atoms)

        loop = ProteinOPESLoop(
            model=model,
            simulator=sim,
            x_a=torch.randn(2, n_atoms * 3),
            x_b=torch.randn(2, n_atoms * 3),
            n_sim_steps=10,
            collect_interval=5,
            kernel_deposit_interval=5,
            max_buffer_size=3,
            bias_type="umbrella",
        )

        for _ in range(5):
            configs, _ = loop.collect_samples()
        # With max_buffer_size=3, oldest chunks are dropped; each chunk
        # is an atomic unit so the actual size may slightly exceed the cap
        assert configs.shape[0] <= 6

    def test_diagnostics(self):
        n_atoms = 3
        model = CommittorMLP(input_dim=n_atoms * 3, hidden_dims=[16])
        sim = _make_mock_openmm_simulator(n_atoms=n_atoms)

        loop = ProteinOPESLoop(
            model=model,
            simulator=sim,
            x_a=torch.randn(2, n_atoms * 3),
            x_b=torch.randn(2, n_atoms * 3),
            n_sim_steps=5,
            bias_type="umbrella",
        )

        diag = loop.diagnostics
        assert "n_kernels" in diag
        assert "rct" in diag
        assert "ess" in diag
        assert "n_accumulated" in diag

    def test_update_bias_no_crash(self):
        """update_bias should not crash even without openmm-torch."""
        n_atoms = 3
        model = CommittorMLP(input_dim=n_atoms * 3, hidden_dims=[16])
        sim = _make_mock_openmm_simulator(n_atoms=n_atoms)

        loop = ProteinOPESLoop(
            model=model,
            simulator=sim,
            x_a=torch.randn(2, n_atoms * 3),
            x_b=torch.randn(2, n_atoms * 3),
            bias_type="umbrella",
        )
        loop.update_bias()

    def test_opes_state_dict_round_trip(self):
        """opes_state_dict -> load_opes_state_dict reproduces state."""
        n_atoms = 3
        model = CommittorMLP(input_dim=n_atoms * 3, hidden_dims=[16])
        sim = _make_mock_openmm_simulator(n_atoms=n_atoms)

        loop = ProteinOPESLoop(
            model=model,
            simulator=sim,
            x_a=torch.randn(2, n_atoms * 3),
            x_b=torch.randn(2, n_atoms * 3),
            n_sim_steps=10,
            collect_interval=5,
            kernel_deposit_interval=1,
            bias_type="umbrella",
        )
        loop.collect_samples()
        state = loop.opes_state_dict()

        loop2 = ProteinOPESLoop(
            model=model,
            simulator=_make_mock_openmm_simulator(n_atoms=n_atoms),
            x_a=torch.randn(2, n_atoms * 3),
            x_b=torch.randn(2, n_atoms * 3),
            bias_type="umbrella",
        )
        assert loop2.opes.n_kernels == 0
        loop2.load_opes_state_dict(state)
        assert loop2.opes.n_kernels == loop.opes.n_kernels
        assert len(loop2._config_buffer) == len(loop._config_buffer)

    def test_combined_vk_opes_bias_type(self):
        """combined_vk_opes should be accepted as bias_type."""
        n_atoms = 3
        model = CommittorMLP(input_dim=n_atoms * 3, hidden_dims=[16])
        sim = _make_mock_openmm_simulator(n_atoms=n_atoms)

        # V_K tracing hits JIT limits in unit tests; only check loop wiring.
        with patch(
            "committorch.simulators.openmm_sim.OpenMMSimulator.add_committor_bias"
        ) as _mock_bias:
            loop = ProteinOPESLoop(
                model=model,
                simulator=sim,
                x_a=torch.randn(2, n_atoms * 3),
                x_b=torch.randn(2, n_atoms * 3),
                bias_type="combined_vk_opes",
            )
        assert loop._bias_type == "combined_vk_opes"
        assert loop.combined is not None
        _mock_bias.assert_called()


class TestProteinUmbrellaLoop:
    def test_construction(self):
        n_atoms = 5
        model = CommittorMLP(input_dim=n_atoms * 3, hidden_dims=[16])
        sim = _make_mock_openmm_simulator(n_atoms=n_atoms)

        x_a = torch.randn(3, n_atoms * 3)
        x_b = torch.randn(3, n_atoms * 3)

        loop = ProteinUmbrellaLoop(
            model=model,
            simulator=sim,
            x_a=x_a,
            x_b=x_b,
            n_windows=5,
            spring_constant=50.0,
        )

        assert loop.n_windows == 5
        assert loop.spring_constant == 50.0
        assert len(loop.targets) == 5

    def test_collect_samples_from_windows(self):
        n_atoms = 3
        model = CommittorMLP(input_dim=n_atoms * 3, hidden_dims=[16])
        sim = _make_mock_openmm_simulator(n_atoms=n_atoms)

        x_a = torch.randn(2, n_atoms * 3)
        x_b = torch.randn(2, n_atoms * 3)

        loop = ProteinUmbrellaLoop(
            model=model,
            simulator=sim,
            x_a=x_a,
            x_b=x_b,
            n_windows=3,
            n_sim_steps=10,
        )

        configs, weights = loop.collect_samples()
        assert configs.shape[0] == 3
        assert weights is None

    def test_umbrella_targets_span_zero_to_one(self):
        n_atoms = 3
        model = CommittorMLP(input_dim=n_atoms * 3, hidden_dims=[16])
        sim = _make_mock_openmm_simulator(n_atoms=n_atoms)

        loop = ProteinUmbrellaLoop(
            model=model,
            simulator=sim,
            x_a=torch.randn(2, n_atoms * 3),
            x_b=torch.randn(2, n_atoms * 3),
            n_windows=11,
        )

        assert loop.targets[0].item() == pytest.approx(0.0)
        assert loop.targets[-1].item() == pytest.approx(1.0)
