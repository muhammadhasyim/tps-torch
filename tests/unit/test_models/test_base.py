"""Tests for CommittorModel base class and interface contract."""

import torch
import pytest

from committorch.models.base import CommittorModel


class DummyCommittor(CommittorModel):
    """Trivial committor for testing the ABC: q(x) = sigmoid(sum(x))."""

    def _latent(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=-1, keepdim=True)


class TestCommittorModelInterface:
    def test_forward_returns_values_in_01(self, default_dtype):
        model = DummyCommittor()
        x = torch.randn(10, 3)
        q = model(x)
        assert q.shape == (10,)
        assert (q >= 0).all() and (q <= 1).all()

    def test_latent_returns_unbounded_scalar(self, default_dtype):
        model = DummyCommittor()
        x = torch.randn(10, 3)
        z = model.latent(x)
        assert z.shape == (10,)

    def test_sigmoid_steepness(self, default_dtype):
        model_p1 = DummyCommittor(sigmoid_steepness=1.0)
        model_p3 = DummyCommittor(sigmoid_steepness=3.0)
        x = torch.tensor([[1.0, 0.0, 0.0]])
        q1 = model_p1(x)
        q3 = model_p3(x)
        assert q3 > q1, "Higher steepness should give sharper sigmoid"

    def test_gradient_shape_matches_input(self, default_dtype):
        model = DummyCommittor()
        x = torch.randn(5, 3)
        grad = model.gradient(x)
        assert grad.shape == x.shape

    def test_gradient_finite_difference(self, default_dtype):
        model = DummyCommittor()
        x = torch.randn(1, 3)
        grad = model.gradient(x)
        eps = 1e-5
        for i in range(3):
            x_plus = x.clone()
            x_minus = x.clone()
            x_plus[0, i] += eps
            x_minus[0, i] -= eps
            fd = (model(x_plus) - model(x_minus)) / (2 * eps)
            torch.testing.assert_close(grad[0, i], fd.squeeze(), atol=1e-4, rtol=1e-4)

    def test_latent_gradient_finite_difference(self, default_dtype):
        model = DummyCommittor()
        x = torch.randn(1, 3)
        grad_z = model.latent_gradient(x)
        eps = 1e-5
        for i in range(3):
            x_plus = x.clone()
            x_minus = x.clone()
            x_plus[0, i] += eps
            x_minus[0, i] -= eps
            fd = (model.latent(x_plus) - model.latent(x_minus)) / (2 * eps)
            torch.testing.assert_close(
                grad_z[0, i], fd.squeeze(), atol=1e-4, rtol=1e-4
            )

    def test_forward_at_origin_is_half(self, default_dtype):
        model = DummyCommittor()
        x = torch.zeros(1, 3)
        q = model(x)
        torch.testing.assert_close(q, torch.tensor([0.5]))
