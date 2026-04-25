"""Tests for OPESBiasModule (TorchForce-traceable V_OPES on z(x))."""

import torch
import pytest

from committorch.models.mlp import CommittorMLP
from committorch.sampling.opes import OPESBias
from committorch.simulators.openmm_sim import OPESBiasModule


class TestOPESBiasModule:
    def _make_components(self, input_dim=6):
        model = CommittorMLP(input_dim=input_dim, hidden_dims=[16])
        opes = OPESBias(
            model, sigma=0.5, barrier=5.0, bias_factor=10.0,
            beta=1.0, compression_threshold=1.0,
        )
        return model, opes

    def test_matches_python_opes_with_kernels(self):
        """Module output matches OPESBias.bias_energy for same inputs."""
        model, opes = self._make_components()
        opes.deposit_kernel(0.0)
        opes.deposit_kernel(0.5)
        opes.deposit_kernel(-0.3)

        opes_state = opes.state_dict()
        module = OPESBiasModule(
            model=model, beta=1.0, opes_state=opes_state,
        )
        module.eval()

        pos = torch.randn(2, 3)
        v_module = module(pos)
        v_python = opes.bias_energy(pos.reshape(1, -1)).squeeze()
        torch.testing.assert_close(v_module, v_python, atol=1e-5, rtol=1e-4)

    def test_empty_kernels_returns_constant(self):
        """With no OPES kernels, V_OPES is a constant floor."""
        model, opes = self._make_components()
        opes_state = opes.state_dict()
        module = OPESBiasModule(
            model=model, beta=1.0, opes_state=opes_state,
        )
        module.eval()
        pos = torch.randn(2, 3)
        v = module(pos)
        assert torch.isfinite(v), f"Got non-finite value: {v}"

    def test_returns_scalar(self):
        """forward() must return a scalar for TorchForce."""
        model, opes = self._make_components()
        opes.deposit_kernel(0.0)
        module = OPESBiasModule(
            model=model, beta=1.0, opes_state=opes.state_dict(),
        )
        module.eval()
        pos = torch.randn(2, 3)
        v = module(pos)
        assert v.dim() == 0, f"Expected scalar, got shape {v.shape}"

    def test_jit_trace_and_reload(self, tmp_path):
        """Module survives torch.jit.trace -> save -> load -> eval."""
        model, opes = self._make_components()
        opes.deposit_kernel(0.0)
        opes.deposit_kernel(1.0)

        module = OPESBiasModule(
            model=model, beta=1.0, opes_state=opes.state_dict(),
        )
        module.eval()

        pos = torch.randn(2, 3)
        v_original = module(pos)

        with torch.no_grad():
            traced = torch.jit.trace(module, pos)
        pt_path = str(tmp_path / "opes.pt")
        traced.save(pt_path)

        loaded = torch.jit.load(pt_path)
        v_loaded = loaded(pos)
        torch.testing.assert_close(v_original, v_loaded, atol=1e-5, rtol=1e-4)

    def test_jit_trace_empty_then_with_kernels(self, tmp_path):
        """Re-tracing with updated kernel state produces different bias."""
        model, opes = self._make_components()

        module_empty = OPESBiasModule(
            model=model, beta=1.0, opes_state=opes.state_dict(),
        )
        module_empty.eval()
        pos = torch.randn(2, 3)
        v_empty = module_empty(pos).item()

        opes.deposit_kernel(0.0)
        opes.deposit_kernel(0.5)
        module_full = OPESBiasModule(
            model=model, beta=1.0, opes_state=opes.state_dict(),
        )
        module_full.eval()
        v_full = module_full(pos).item()

        assert v_empty != pytest.approx(v_full, abs=1e-6), (
            "Bias should change when kernels are added"
        )

    def test_gradient_flows_through_positions(self):
        """TorchForce needs dV/dpositions; verify autograd works."""
        model, opes = self._make_components()
        opes.deposit_kernel(0.0)

        module = OPESBiasModule(
            model=model, beta=1.0, opes_state=opes.state_dict(),
        )
        module.eval()

        pos = torch.randn(2, 3, requires_grad=True)
        v = module(pos)
        v.backward()
        assert pos.grad is not None
        assert torch.isfinite(pos.grad).all()
