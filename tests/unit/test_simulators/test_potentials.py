"""Tests for analytical potential energy surfaces."""

import torch
import pytest

from committorch.simulators.potentials import muller_brown, quartic_1d


class TestMullerBrown:
    def test_energy_shape(self, default_dtype):
        xy = torch.randn(10, 2)
        v = muller_brown.energy(xy)
        assert v.shape == (10,)

    def test_energy_single_point(self, default_dtype):
        xy = torch.zeros(1, 2)
        v = muller_brown.energy(xy)
        assert v.shape == (1,)
        assert torch.isfinite(v).all()

    def test_gradient_shape(self, default_dtype):
        xy = torch.randn(10, 2)
        g = muller_brown.gradient(xy)
        assert g.shape == (10, 2)

    def test_gradient_vs_autograd(self, default_dtype):
        xy = torch.randn(20, 2, requires_grad=True)
        v = muller_brown.energy(xy)
        (auto_grad,) = torch.autograd.grad(v.sum(), xy)
        analytic_grad = muller_brown.gradient(xy.detach())
        torch.testing.assert_close(auto_grad, analytic_grad, atol=1e-10, rtol=1e-10)

    def test_minima_are_low_energy(self, default_dtype):
        mins = torch.stack([
            muller_brown.REACTANT_MINIMUM,
            muller_brown.PRODUCT_MINIMUM,
            muller_brown.INTERMEDIATE_MINIMUM,
        ])
        energies = muller_brown.energy(mins)
        for i, e in enumerate(energies):
            assert e < 0, f"Minimum {i} should have negative energy, got {e}"

    def test_gradient_near_zero_at_minima(self, default_dtype):
        for name, m in [
            ("reactant", muller_brown.REACTANT_MINIMUM),
            ("product", muller_brown.PRODUCT_MINIMUM),
        ]:
            g = muller_brown.gradient(m.unsqueeze(0))
            assert g.norm() < 5.0, f"Gradient at {name} minimum should be small"


class TestQuartic1D:
    def test_energy_at_wells(self, default_dtype):
        x = torch.tensor([-1.0, 1.0])
        v = quartic_1d.energy(x)
        torch.testing.assert_close(v, torch.zeros(2))

    def test_energy_at_barrier(self, default_dtype):
        x = torch.tensor([0.0])
        v = quartic_1d.energy(x, barrier_height=2.0)
        torch.testing.assert_close(v, torch.tensor([2.0]))

    def test_gradient_at_wells_is_zero(self, default_dtype):
        x = torch.tensor([-1.0, 1.0])
        g = quartic_1d.gradient(x)
        torch.testing.assert_close(g, torch.zeros(2), atol=1e-14, rtol=0.0)

    def test_gradient_vs_finite_difference(self, default_dtype):
        x = torch.linspace(-1.5, 1.5, 50)
        g = quartic_1d.gradient(x)
        eps = 1e-7
        fd = (quartic_1d.energy(x + eps) - quartic_1d.energy(x - eps)) / (2 * eps)
        torch.testing.assert_close(g, fd, atol=1e-5, rtol=1e-5)

    def test_exact_committor_boundaries(self, default_dtype):
        x_a = torch.tensor([-1.0])
        x_b = torch.tensor([1.0])
        q_a = quartic_1d.exact_committor(x_a, kT=1.0)
        q_b = quartic_1d.exact_committor(x_b, kT=1.0)
        torch.testing.assert_close(q_a, torch.tensor([0.0]), atol=1e-3, rtol=0.0)
        torch.testing.assert_close(q_b, torch.tensor([1.0]), atol=1e-3, rtol=0.0)

    def test_exact_committor_monotonic(self, default_dtype):
        x = torch.linspace(-1.0, 1.0, 100)
        q = quartic_1d.exact_committor(x, kT=1.0)
        diffs = q[1:] - q[:-1]
        assert (diffs >= -1e-6).all(), "Committor should be monotonically increasing"

    def test_exact_committor_symmetric(self, default_dtype):
        x = torch.tensor([0.0])
        q = quartic_1d.exact_committor(x, kT=1.0)
        torch.testing.assert_close(q, torch.tensor([0.5]), atol=1e-2, rtol=0.0)
