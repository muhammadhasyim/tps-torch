"""Tests for committor training loss functions."""

import torch
import pytest

from committorch.models.mlp import CommittorMLP
from committorch.training.losses import BKELoss, BoundaryLoss, SupervisedLoss, CombinedLoss


class TestBKELoss:
    def test_returns_scalar(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        loss_fn = BKELoss(model)
        x = torch.randn(20, 2)
        loss = loss_fn(x)
        assert loss.shape == ()
        assert loss >= 0

    def test_nonnegative(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        loss_fn = BKELoss(model)
        for _ in range(10):
            x = torch.randn(20, 2)
            assert loss_fn(x) >= 0

    def test_with_weights(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        loss_fn = BKELoss(model)
        x = torch.randn(20, 2)
        w = torch.ones(20)
        loss_w = loss_fn(x, weights=w)
        loss_no_w = loss_fn(x)
        torch.testing.assert_close(loss_w, loss_no_w, atol=1e-10, rtol=1e-10)

    def test_backpropagates(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        loss_fn = BKELoss(model)
        model.train()
        x = torch.randn(10, 2)
        loss = loss_fn(x)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


class TestBoundaryLoss:
    def test_zero_when_perfect(self, default_dtype):
        """If q(x_a)=0 and q(x_b)=1, boundary loss should be zero."""
        model = CommittorMLP(input_dim=1, hidden_dims=[16])
        loss_fn = BoundaryLoss(model)
        x_a = torch.tensor([[-100.0]])
        x_b = torch.tensor([[100.0]])
        q_a = model(x_a).item()
        q_b = model(x_b).item()
        loss = loss_fn(x_a, x_b)
        expected = q_a**2 + (q_b - 1.0)**2
        torch.testing.assert_close(loss, torch.tensor(expected), atol=1e-6, rtol=1e-6)

    def test_returns_scalar(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        loss_fn = BoundaryLoss(model)
        x_a = torch.randn(10, 2)
        x_b = torch.randn(10, 2)
        loss = loss_fn(x_a, x_b)
        assert loss.shape == ()

    def test_nonnegative(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        loss_fn = BoundaryLoss(model)
        x_a = torch.randn(10, 2)
        x_b = torch.randn(10, 2)
        assert loss_fn(x_a, x_b) >= 0


class TestSupervisedLoss:
    def test_zero_when_perfect(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        loss_fn = SupervisedLoss(model)
        x = torch.randn(10, 2)
        q_pred = model(x).detach()
        loss = loss_fn(x, q_pred)
        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-10, rtol=0.0)

    def test_positive_when_imperfect(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        loss_fn = SupervisedLoss(model)
        x = torch.randn(10, 2)
        q_wrong = torch.ones(10)
        assert loss_fn(x, q_wrong) > 0


class TestCombinedLoss:
    def test_returns_scalar(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        loss_fn = CombinedLoss(model, lambda_boundary=1.0)
        x = torch.randn(20, 2)
        x_a = torch.randn(5, 2)
        x_b = torch.randn(5, 2)
        loss = loss_fn(x, x_a, x_b)
        assert loss.shape == ()

    def test_with_supervised(self, default_dtype):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        loss_fn = CombinedLoss(model, lambda_boundary=1.0, lambda_supervised=0.5)
        x = torch.randn(20, 2)
        x_a = torch.randn(5, 2)
        x_b = torch.randn(5, 2)
        x_s = torch.randn(10, 2)
        q_s = torch.rand(10)
        loss = loss_fn(x, x_a, x_b, x_supervised=x_s, q_empirical=q_s)
        assert loss.shape == ()
        assert loss > 0
