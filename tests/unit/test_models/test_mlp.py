"""Tests for the CommittorMLP model."""

import torch
import pytest

from committorch.models.mlp import CommittorMLP


class TestCommittorMLP:
    def test_output_range(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[32, 16])
        x = torch.randn(50, 2)
        q = model(x)
        assert q.shape == (50,)
        assert (q > 0).all() and (q < 1).all()

    def test_single_hidden_layer_matches_paper(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[200], activation="ReLU")
        assert len(list(model.net.children())) == 3  # Linear, ReLU, Linear

    def test_multi_layer(self, default_dtype):
        model = CommittorMLP(
            input_dim=3, hidden_dims=[64, 32, 16], activation="Tanh"
        )
        x = torch.randn(10, 3)
        q = model(x)
        assert q.shape == (10,)

    def test_gradient_is_differentiable(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[32])
        model.train()
        x = torch.randn(5, 2)
        grad_q = model.gradient(x)
        loss = (grad_q**2).sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_latent_is_pre_sigmoid(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[32])
        x = torch.randn(5, 2)
        z = model.latent(x)
        q = model(x)
        q_from_z = torch.sigmoid(model.sigmoid_steepness * z)
        torch.testing.assert_close(q, q_from_z)

    def test_gradient_finite_difference(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[32])
        model.eval()
        x = torch.randn(1, 2)
        grad = model.gradient(x)
        eps = 1e-5
        for i in range(2):
            x_p = x.clone()
            x_m = x.clone()
            x_p[0, i] += eps
            x_m[0, i] -= eps
            fd = (model(x_p) - model(x_m)) / (2 * eps)
            torch.testing.assert_close(
                grad[0, i], fd.squeeze(), atol=1e-4, rtol=1e-4
            )
