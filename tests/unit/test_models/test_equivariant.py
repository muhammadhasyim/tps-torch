"""Tests for equivariant models and the pretrained backbone interface."""

import torch
import pytest

from committorch.models.equivariant.backbone import PretrainedBackbone, DistanceEncoder
from committorch.models.equivariant.mace import MACECommittor
from committorch.models.equivariant.schnet import SchNetCommittor
from committorch.models.equivariant.painn import PaiNNCommittor
from committorch.models.pretrained import load_pretrained


class TestDistanceEncoder:
    def test_output_shape(self):
        enc = DistanceEncoder(n_atoms=5, hidden_dim=32, output_dim=16)
        pos = torch.randn(3, 5, 3)
        out = enc(pos)
        assert out.shape == (3, 16)

    def test_se3_invariance_rotation(self):
        """Pairwise distances are invariant under rotation."""
        enc = DistanceEncoder(n_atoms=4, hidden_dim=32, output_dim=16)
        pos = torch.randn(2, 4, 3)

        theta = torch.tensor(0.7)
        R = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1],
        ], dtype=pos.dtype)

        pos_rot = pos @ R.T
        out_orig = enc(pos)
        out_rot = enc(pos_rot)
        torch.testing.assert_close(out_orig, out_rot, atol=1e-5, rtol=1e-5)

    def test_se3_invariance_translation(self):
        enc = DistanceEncoder(n_atoms=4, hidden_dim=32, output_dim=16)
        pos = torch.randn(2, 4, 3)
        shift = torch.randn(1, 1, 3)
        out_orig = enc(pos)
        out_shift = enc(pos + shift)
        torch.testing.assert_close(out_orig, out_shift, atol=1e-5, rtol=1e-5)

    def test_flat_input(self):
        enc = DistanceEncoder(n_atoms=3, hidden_dim=16, output_dim=8)
        pos_flat = torch.randn(5, 9)
        out = enc(pos_flat)
        assert out.shape == (5, 8)


class TestPretrainedBackbone:
    def test_committor_output_range(self):
        enc = DistanceEncoder(n_atoms=5, output_dim=16)
        model = PretrainedBackbone(enc, feature_dim=16)
        pos = torch.randn(10, 5, 3)
        q = model(pos)
        assert q.shape == (10,)
        assert (q > 0).all() and (q < 1).all()

    def test_freeze_encoder(self):
        enc = DistanceEncoder(n_atoms=5, output_dim=16)
        model = PretrainedBackbone(enc, feature_dim=16, freeze_encoder=True)
        for p in model.encoder.parameters():
            assert not p.requires_grad
        for p in model.head.parameters():
            assert p.requires_grad

    def test_unfreeze_encoder(self):
        enc = DistanceEncoder(n_atoms=5, output_dim=16)
        model = PretrainedBackbone(enc, feature_dim=16, freeze_encoder=True)
        groups = model.unfreeze_encoder(lr_scale=0.1)
        assert len(groups) == 2
        for p in model.encoder.parameters():
            assert p.requires_grad

    def test_gradient_works(self):
        enc = DistanceEncoder(n_atoms=3, output_dim=8)
        model = PretrainedBackbone(enc, feature_dim=8, freeze_encoder=False)
        pos = torch.randn(2, 3, 3)
        grad = model.gradient(pos)
        assert grad.shape == pos.shape

    def test_se3_invariance_committor(self):
        """Full committor q(x) should be SE(3)-invariant."""
        enc = DistanceEncoder(n_atoms=4, output_dim=16)
        model = PretrainedBackbone(enc, feature_dim=16, freeze_encoder=False)
        model.eval()
        pos = torch.randn(3, 4, 3)

        theta = torch.tensor(1.2)
        R = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1],
        ], dtype=pos.dtype)

        with torch.no_grad():
            q_orig = model(pos)
            q_rot = model(pos @ R.T)
            q_trans = model(pos + torch.randn(1, 1, 3))

        torch.testing.assert_close(q_orig, q_rot, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(q_orig, q_trans, atol=1e-5, rtol=1e-5)


class TestPlaceholderModels:
    """Test that placeholder MACE/SchNet/PaiNN models work without deps."""

    def test_mace_placeholder(self):
        model = MACECommittor(mace_model=None, feature_dim=32)
        x = torch.randn(5, 10)
        q = model(x)
        assert q.shape == (5,)

    def test_schnet_placeholder(self):
        model = SchNetCommittor(schnet_model=None, feature_dim=32)
        x = torch.randn(5, 10)
        q = model(x)
        assert q.shape == (5,)

    def test_painn_placeholder(self):
        model = PaiNNCommittor(painn_model=None, feature_dim=32)
        x = torch.randn(5, 10)
        q = model(x)
        assert q.shape == (5,)


class TestPretrainedRegistry:
    def test_load_distance_encoder(self):
        model = load_pretrained("distance_encoder", n_atoms=5, freeze_encoder=True)
        pos = torch.randn(3, 5, 3)
        q = model(pos)
        assert q.shape == (3,)

    def test_load_mace_placeholder(self):
        model = load_pretrained("mace", freeze_encoder=True)
        x = torch.randn(3, 10)
        q = model(x)
        assert q.shape == (3,)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown model name"):
            load_pretrained("nonexistent_model")
