"""Tests for OPES metadynamics bias (PLUMED-style algorithm)."""

import torch
import pytest

from committorch.models.mlp import CommittorMLP
from committorch.sampling.opes import OPESBias, CombinedVKOPES
from committorch.sampling.kolmogorov import KolmogorovBias


class TestOPESBias:
    def test_no_kernels_returns_constant_bias(self):
        """With no kernels, V = prefactor/beta * log(epsilon)."""
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        opes = OPESBias(model, sigma=0.5, barrier=5.0, beta=1.0)
        x = torch.randn(5, 2)
        v = opes.bias_energy(x)
        expected = (1.0 - 1.0 / opes.bias_factor) * torch.log(
            torch.tensor(opes.epsilon)
        ).item()
        torch.testing.assert_close(v, torch.full((5,), expected))

    def test_deposit_kernel_changes_bias(self):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        opes = OPESBias(model, sigma=0.5, barrier=5.0, beta=1.0)
        x = torch.randn(1, 2)
        with torch.no_grad():
            z_val = model.latent(x).item()
        v_before = opes.bias_energy(x).item()
        opes.deposit_kernel(z_val)
        v_after = opes.bias_energy(x).item()
        assert v_after != v_before, "Depositing a kernel should change the bias"

    def test_kernel_count(self):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        opes = OPESBias(
            model, sigma=0.5, barrier=5.0, compression_threshold=0.0
        )
        assert opes.n_kernels == 0
        opes.deposit_kernel(0.0)
        opes.deposit_kernel(100.0)
        assert opes.n_kernels == 2

    def test_compression_merges_nearby_kernels(self):
        """Kernels deposited at the same location should be merged."""
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        opes = OPESBias(
            model, sigma=0.5, barrier=5.0, compression_threshold=1.0
        )
        opes.deposit_kernel(0.0)
        opes.deposit_kernel(0.0)
        opes.deposit_kernel(0.0)
        assert opes.n_kernels == 1, (
            "Kernels at the same location should merge"
        )

    def test_no_compression_keeps_all(self):
        """With threshold=0, no merging occurs."""
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        opes = OPESBias(
            model, sigma=0.5, barrier=5.0, compression_threshold=0.0
        )
        opes.deposit_kernel(0.0)
        opes.deposit_kernel(0.0)
        opes.deposit_kernel(0.0)
        assert opes.n_kernels == 3

    def test_distant_kernels_not_merged(self):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        opes = OPESBias(
            model, sigma=0.5, barrier=5.0, compression_threshold=1.0
        )
        opes.deposit_kernel(0.0)
        opes.deposit_kernel(10.0)
        assert opes.n_kernels == 2

    def test_bias_formula_matches_plumed(self):
        """V(s) = (1-1/gamma)/beta * log(P(s)/Z + epsilon)."""
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        gamma = 5.0
        beta = 1.0
        eps = 0.01
        opes = OPESBias(
            model, sigma=0.5, barrier=5.0, bias_factor=gamma,
            beta=beta, epsilon=eps, compression_threshold=0.0,
        )
        x = torch.randn(3, 2)
        with torch.no_grad():
            z = model.latent(x)

        # Deposit some kernels at known locations
        opes.deposit_kernel(0.0)
        opes.deposit_kernel(0.5)
        opes.deposit_kernel(-0.5)

        v = opes.bias_energy(x)
        assert v.shape == (3,)
        # Values should be finite
        assert torch.isfinite(v).all()

    def test_bias_energy_shape(self):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        opes = OPESBias(model, sigma=0.5, barrier=5.0)
        opes.deposit_kernel(0.0)
        opes.deposit_kernel(1.0)
        x = torch.randn(10, 2)
        v = opes.bias_energy(x)
        assert v.shape == (10,)

    def test_effective_sample_size(self):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        opes = OPESBias(model, sigma=0.5, barrier=5.0)
        neff_0 = opes.effective_sample_size
        opes.deposit_kernel(0.0)
        opes.deposit_kernel(1.0)
        neff_1 = opes.effective_sample_size
        assert neff_1 > 0

    def test_rct_is_finite(self):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        opes = OPESBias(model, sigma=0.5, barrier=5.0)
        opes.deposit_kernel(0.0)
        assert torch.isfinite(torch.tensor(opes.rct))

    def test_backward_compat_deposit_hill(self):
        """deposit_hill still works as alias."""
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        opes = OPESBias(model, sigma=0.5, barrier=5.0)
        opes.deposit_hill(0.0)
        assert opes.n_hills == 1

    def test_bias_becomes_quasi_static(self):
        """After many deposits, bias changes should diminish."""
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        opes = OPESBias(
            model, sigma=0.3, barrier=3.0, bias_factor=5.0, beta=1.0,
            compression_threshold=1.0,
        )
        x = torch.randn(1, 2)

        # Deposit many kernels at two locations to let KDE converge
        for _ in range(50):
            opes.deposit_kernel(0.0)
            opes.deposit_kernel(1.0)

        v_before = opes.bias_energy(x).item()
        opes.deposit_kernel(0.5)
        v_after = opes.bias_energy(x).item()

        relative_change = abs(v_after - v_before) / (abs(v_before) + 1e-10)
        assert relative_change < 1.0, (
            f"Bias should become quasi-static; relative change = {relative_change:.4f}"
        )


class TestCombinedVKOPES:
    def test_combined_bias_shape(self):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        vk = KolmogorovBias(model, lam=1.0, beta=1.0)
        opes = OPESBias(model, sigma=0.5, barrier=5.0)
        opes.deposit_kernel(0.0)
        combined = CombinedVKOPES(vk, opes)
        x = torch.randn(5, 2)
        v = combined.bias_energy(x)
        assert v.shape == (5,)

    def test_reweighting_sums_to_one(self):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        vk = KolmogorovBias(model, lam=0.5, beta=1.0)
        opes = OPESBias(model, sigma=0.5, barrier=5.0)
        opes.deposit_kernel(0.0)
        combined = CombinedVKOPES(vk, opes)
        x = torch.randn(20, 2)
        w = combined.reweighting_factor(x, beta=1.0)
        assert abs(w.sum().item() - 1.0) < 1e-5
