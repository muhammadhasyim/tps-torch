"""Tests for CommittorDataset."""

import torch
import pytest

from committorch.data.datasets import CommittorDataset


class TestCommittorDataset:
    def test_len(self):
        ds = CommittorDataset(torch.randn(100, 10))
        assert len(ds) == 100

    def test_getitem(self):
        ds = CommittorDataset(
            configs=torch.randn(50, 6),
            labels=torch.rand(50),
            weights=torch.ones(50),
        )
        item = ds[0]
        assert "configs" in item
        assert "labels" in item
        assert "weights" in item
        assert item["configs"].shape == (6,)

    def test_no_labels(self):
        ds = CommittorDataset(torch.randn(20, 4))
        item = ds[0]
        assert "labels" not in item

    def test_from_trajectory(self):
        pos = torch.randn(30, 5, 3)
        ds = CommittorDataset.from_trajectory(pos, flatten=True)
        assert len(ds) == 30
        assert ds[0]["configs"].shape == (15,)

    def test_from_trajectory_no_flatten(self):
        pos = torch.randn(30, 5, 3)
        ds = CommittorDataset.from_trajectory(pos, flatten=False)
        assert ds[0]["configs"].shape == (5, 3)
