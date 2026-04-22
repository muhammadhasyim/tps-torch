"""Tests for reaction rate computation."""

import torch
import pytest

from committorch.models.mlp import CommittorMLP
from committorch.analysis.rates import rate_from_committor_gradient, rate_from_bke_loss


class TestRateFromCommittorGradient:
    def test_returns_scalar(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        configs = torch.randn(50, 2)
        rate = rate_from_committor_gradient(model, configs, kT=1.0, friction=1.0)
        assert rate.shape == ()

    def test_nonnegative(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        configs = torch.randn(50, 2)
        rate = rate_from_committor_gradient(model, configs, kT=1.0, friction=1.0)
        assert rate >= 0

    def test_scales_with_kT(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        configs = torch.randn(50, 2)
        r1 = rate_from_committor_gradient(model, configs, kT=1.0, friction=1.0)
        r2 = rate_from_committor_gradient(model, configs, kT=2.0, friction=1.0)
        torch.testing.assert_close(r2, 2.0 * r1, atol=1e-10, rtol=1e-10)

    def test_with_weights(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        configs = torch.randn(50, 2)
        w = torch.ones(50)
        r_w = rate_from_committor_gradient(model, configs, kT=1.0, friction=1.0, weights=w)
        r_no_w = rate_from_committor_gradient(model, configs, kT=1.0, friction=1.0)
        torch.testing.assert_close(r_w, r_no_w, atol=1e-8, rtol=1e-8)


class TestRateFromBKELoss:
    def test_basic_conversion(self):
        assert rate_from_bke_loss(1.0, kT=1.0, friction=1.0) == 1.0
        assert rate_from_bke_loss(2.0, kT=1.0, friction=1.0) == 2.0
        assert rate_from_bke_loss(1.0, kT=2.0, friction=1.0) == 2.0
        assert rate_from_bke_loss(1.0, kT=1.0, friction=2.0) == 0.5
