"""Tests for importance-sampling reweighting."""

import torch
import pytest

from committorch.training.reweighting import fep_reweighting, master_equation_weights


class TestFEPReweighting:
    def test_returns_normalized_weights(self, default_dtype):
        configs = [torch.randn(20, 2) for _ in range(5)]
        biases = [torch.rand(20) for _ in range(5)]
        z = fep_reweighting(configs, biases, kT=1.0)
        assert len(z) == 5
        total = sum(zi.item() for zi in z)
        assert abs(total - 1.0) < 1e-6

    def test_uniform_bias_gives_equal_weights(self, default_dtype):
        configs = [torch.randn(20, 2) for _ in range(3)]
        biases = [torch.zeros(20) for _ in range(3)]
        z = fep_reweighting(configs, biases, kT=1.0)
        for zi in z:
            assert abs(zi.item() - 1.0 / 3.0) < 1e-4


class TestMasterEquationWeights:
    def test_returns_normalized(self, default_dtype):
        crossing = torch.tensor([
            [0.0, 5.0, 0.0],
            [5.0, 0.0, 5.0],
            [0.0, 5.0, 0.0],
        ])
        n_steps = torch.tensor([100.0, 100.0, 100.0])
        z = master_equation_weights(crossing, n_steps)
        assert abs(z.sum().item() - 1.0) < 1e-6

    def test_symmetric_system_gives_equal_weights(self, default_dtype):
        crossing = torch.tensor([
            [0.0, 10.0],
            [10.0, 0.0],
        ])
        n_steps = torch.tensor([100.0, 100.0])
        z = master_equation_weights(crossing, n_steps)
        torch.testing.assert_close(z[0], z[1], atol=1e-5, rtol=1e-5)
