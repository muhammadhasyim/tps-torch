"""Tests for feature extraction utilities."""

import torch
import pytest

from committorch.data.features import pairwise_distances, contact_map, rmsd, calpha_distances


class TestPairwiseDistances:
    def test_shape_batch(self):
        pos = torch.randn(5, 4, 3)
        d = pairwise_distances(pos)
        assert d.shape == (5, 6)

    def test_shape_single(self):
        pos = torch.randn(4, 3)
        d = pairwise_distances(pos)
        assert d.shape == (6,)

    def test_nonnegative(self):
        pos = torch.randn(3, 5, 3)
        d = pairwise_distances(pos)
        assert (d >= 0).all()

    def test_zero_for_same_positions(self):
        pos = torch.zeros(2, 3, 3)
        d = pairwise_distances(pos)
        assert d.max().item() < 1e-4

    def test_specific_pairs(self):
        pos = torch.tensor([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ])
        pairs = torch.tensor([[0, 1], [0, 2]])
        d = pairwise_distances(pos, atom_pairs=pairs)
        assert d.shape == (1, 2)
        torch.testing.assert_close(d[0, 0], torch.tensor(1.0), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(d[0, 1], torch.tensor(1.0), atol=1e-4, rtol=1e-4)


class TestContactMap:
    def test_shape(self):
        pos = torch.randn(5, 4, 3)
        c = contact_map(pos, cutoff=10.0)
        assert c.shape == (5, 6)

    def test_binary(self):
        pos = torch.randn(5, 4, 3)
        c = contact_map(pos, cutoff=10.0)
        assert ((c == 0) | (c == 1)).all()


class TestRMSD:
    def test_zero_for_identical(self):
        pos = torch.randn(3, 3)
        r = rmsd(pos, pos)
        assert r.item() < 1e-5

    def test_batch(self):
        ref = torch.randn(4, 3)
        pos = torch.randn(10, 4, 3)
        r = rmsd(pos, ref)
        assert r.shape == (10,)
        assert (r >= 0).all()

    def test_translation_invariant(self):
        ref = torch.randn(4, 3)
        pos = ref.unsqueeze(0) + torch.randn(1, 1, 3)
        r = rmsd(pos, ref)
        assert r.item() < 1e-5


class TestCalphaDistances:
    def test_shape(self):
        pos = torch.randn(5, 10, 3)
        ca_idx = [0, 3, 5, 8]
        d = calpha_distances(pos, ca_idx)
        assert d.shape == (5, 6)
