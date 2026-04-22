"""Tests for Kolmogorov bias V_K."""

import torch
import pytest

from committorch.models.mlp import CommittorMLP
from committorch.sampling.kolmogorov import KolmogorovBias


class TestKolmogorovBias:
    def test_bias_energy_shape(self):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        vk = KolmogorovBias(model, lam=1.0, beta=1.0)
        x = torch.randn(10, 2)
        v = vk.bias_energy(x)
        assert v.shape == (10,)

    def test_bias_is_finite(self):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        vk = KolmogorovBias(model, lam=1.0, beta=1.0)
        x = torch.randn(20, 2)
        v = vk.bias_energy(x)
        assert torch.isfinite(v).all()

    def test_stronger_lambda_gives_larger_bias(self):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        x = torch.randn(10, 2)
        vk1 = KolmogorovBias(model, lam=0.5, beta=1.0)
        vk2 = KolmogorovBias(model, lam=2.0, beta=1.0)
        v1 = vk1.bias_energy(x).abs().mean()
        v2 = vk2.bias_energy(x).abs().mean()
        assert v2 > v1

    def test_transition_state_density_shape(self):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        vk = KolmogorovBias(model, lam=1.0, beta=1.0)
        x = torch.randn(10, 2)
        pk = vk.transition_state_density(x)
        assert pk.shape == (10,)
        assert (pk >= 0).all()

    def test_as_bias_fn(self):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        vk = KolmogorovBias(model, lam=1.0, beta=1.0)
        fn = vk.as_bias_fn()
        x = torch.randn(5, 2)
        torch.testing.assert_close(fn(x), vk.bias_energy(x))
