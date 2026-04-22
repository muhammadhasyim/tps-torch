"""Tests for the finite-temperature string method."""

import torch
import pytest

from committorch.sampling.fts import FTSString, FTSPathBias


class TestFTSString:
    def test_voronoi_assignment(self, default_dtype):
        nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        string = FTSString(nodes)
        x = torch.tensor([[0.1, 0.0], [0.9, 0.0], [1.8, 0.1]])
        assignments = string.voronoi_assignment(x)
        assert assignments.tolist() == [0, 1, 2]

    def test_update_moves_toward_samples(self, default_dtype):
        nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        string = FTSString(nodes, smoothing=0.0)
        old_node1 = string.nodes[1].clone()

        samples = {1: torch.tensor([[1.5, 0.3], [1.3, 0.1]])}
        string.update(samples, learning_rate=1.0)

        dist_old = (old_node1 - torch.tensor([1.4, 0.2])).norm()
        dist_new = (string.nodes[1] - torch.tensor([1.4, 0.2])).norm()
        assert dist_new < dist_old

    def test_reparametrize_equal_spacing(self, default_dtype):
        nodes = torch.tensor([[0.0, 0.0], [0.1, 0.0], [1.0, 0.0]])
        string = FTSString(nodes)
        string._reparametrize()
        seg1 = (string.nodes[1] - string.nodes[0]).norm()
        seg2 = (string.nodes[2] - string.nodes[1]).norm()
        torch.testing.assert_close(seg1, seg2, atol=1e-5, rtol=1e-5)

    def test_endpoints_preserved(self, default_dtype):
        nodes = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0]])
        string = FTSString(nodes)
        start = string.nodes[0].clone()
        end = string.nodes[-1].clone()
        string._reparametrize()
        torch.testing.assert_close(string.nodes[0], start)
        torch.testing.assert_close(string.nodes[-1], end)


class TestFTSPathBias:
    def test_bias_energy_shape(self, default_dtype):
        nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        string = FTSString(nodes)
        bias = FTSPathBias(string)
        x = torch.randn(10, 2)
        w = bias.bias_energy(x, window_idx=1)
        assert w.shape == (10,)

    def test_bias_zero_at_node(self, default_dtype):
        nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        string = FTSString(nodes)
        bias = FTSPathBias(string)
        x = string.nodes[1].unsqueeze(0)
        w = bias.bias_energy(x, window_idx=1)
        torch.testing.assert_close(w, torch.zeros(1), atol=1e-10, rtol=0.0)

    def test_bias_nonnegative(self, default_dtype):
        nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        string = FTSString(nodes)
        bias = FTSPathBias(string)
        x = torch.randn(20, 2)
        w = bias.bias_energy(x, window_idx=1)
        assert (w >= -1e-10).all()
