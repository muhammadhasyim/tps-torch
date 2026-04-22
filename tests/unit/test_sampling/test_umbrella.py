"""Tests for committor-based umbrella sampling."""

import torch
import pytest

from committorch.models.mlp import CommittorMLP
from committorch.sampling.umbrella import UmbrellaBias, UmbrellaWindows


class TestUmbrellaBias:
    def test_bias_energy_shape(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        bias = UmbrellaBias(model, target=0.5, spring_constant=100.0)
        x = torch.randn(10, 2)
        w = bias.bias_energy(x)
        assert w.shape == (10,)

    def test_bias_minimum_at_target(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        target = 0.5
        bias = UmbrellaBias(model, target=target, spring_constant=100.0)
        x = torch.randn(100, 2)
        q = model(x)
        w = bias.bias_energy(x)
        closest_idx = (q - target).abs().argmin()
        assert w[closest_idx] <= w.median()

    def test_bias_nonnegative(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        bias = UmbrellaBias(model, target=0.3, spring_constant=50.0)
        x = torch.randn(20, 2)
        w = bias.bias_energy(x)
        assert (w >= 0).all()

    def test_as_bias_fn(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        bias = UmbrellaBias(model, target=0.5, spring_constant=100.0)
        fn = bias.as_bias_fn()
        x = torch.randn(5, 2)
        assert torch.allclose(fn(x), bias.bias_energy(x))


class TestUmbrellaWindows:
    def test_window_count(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        windows = UmbrellaWindows(model, n_windows=10)
        assert len(windows) == 10

    def test_targets_span_01(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        windows = UmbrellaWindows(model, n_windows=5)
        assert windows.targets[0] == pytest.approx(0.0)
        assert windows.targets[-1] == pytest.approx(1.0)

    def test_indexing(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        windows = UmbrellaWindows(model, n_windows=5)
        w = windows[2]
        assert isinstance(w, UmbrellaBias)
        assert w.target == pytest.approx(0.5)
