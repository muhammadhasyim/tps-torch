"""Tests for the overdamped Langevin integrator."""

import torch
import pytest

from committorch.simulators.langevin import OverdampedLangevin
from committorch.simulators.potentials import muller_brown, quartic_1d


class TestOverdampedLangevin:
    def test_step_preserves_shape(self, default_dtype):
        sim = OverdampedLangevin(
            dim=2,
            potential_fn=muller_brown.energy,
            gradient_fn=muller_brown.gradient,
            kT=1.0,
            dt=1e-4,
        )
        x = torch.randn(10, 2)
        x_new = sim.step(x, n_steps=5)
        assert x_new.shape == x.shape

    def test_deterministic_with_zero_noise(self, default_dtype):
        sim = OverdampedLangevin(
            dim=1,
            potential_fn=lambda x: quartic_1d.energy(x.squeeze(-1)).unsqueeze(-1),
            gradient_fn=lambda x: quartic_1d.gradient(x.squeeze(-1)).unsqueeze(-1),
            kT=0.0,
            dt=0.01,
        )
        sim._noise_scale = 0.0
        x = torch.tensor([[0.5]])
        x1 = sim.step(x.clone(), n_steps=1)
        x2 = sim.step(x.clone(), n_steps=1)
        torch.testing.assert_close(x1, x2)

    def test_moves_toward_minimum(self, default_dtype):
        sim = OverdampedLangevin(
            dim=1,
            potential_fn=lambda x: quartic_1d.energy(x.squeeze(-1)).unsqueeze(-1),
            gradient_fn=lambda x: quartic_1d.gradient(x.squeeze(-1)).unsqueeze(-1),
            kT=0.0,
            dt=0.001,
        )
        sim._noise_scale = 0.0
        x = torch.tensor([[0.5]])
        for _ in range(1000):
            x = sim.step(x, n_steps=1)
        assert abs(x.item() - 1.0) < 0.1, f"Should converge to well at +1, got {x.item()}"

    def test_bias_modifies_trajectory(self, default_dtype):
        sim = OverdampedLangevin(
            dim=2,
            potential_fn=muller_brown.energy,
            gradient_fn=muller_brown.gradient,
            kT=1.0,
            dt=1e-4,
        )
        torch.manual_seed(42)
        x = torch.zeros(5, 2)
        x_unbiased = sim.step(x.clone(), n_steps=10)

        sim.set_bias(lambda x: 100.0 * (x**2).sum(dim=-1))
        torch.manual_seed(42)
        x_biased = sim.step(x.clone(), n_steps=10)

        assert not torch.allclose(x_unbiased, x_biased)

    def test_potential_energy(self, default_dtype):
        sim = OverdampedLangevin(
            dim=2,
            potential_fn=muller_brown.energy,
            gradient_fn=muller_brown.gradient,
        )
        xy = torch.randn(5, 2)
        v_sim = sim.potential_energy(xy)
        v_direct = muller_brown.energy(xy)
        torch.testing.assert_close(v_sim, v_direct)

    def test_force_is_negative_gradient(self, default_dtype):
        sim = OverdampedLangevin(
            dim=2,
            potential_fn=muller_brown.energy,
            gradient_fn=muller_brown.gradient,
        )
        xy = torch.randn(5, 2)
        f = sim.force(xy)
        g = muller_brown.gradient(xy)
        torch.testing.assert_close(f, -g)
