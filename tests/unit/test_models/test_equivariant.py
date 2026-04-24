"""Tests for equivariant models, graph input, export, and the pretrained backbone."""

import torch
import pytest
from pathlib import Path

from committorch.models.equivariant.backbone import PretrainedBackbone, DistanceEncoder
from committorch.models.equivariant.mace import MACECommittor
from committorch.models.equivariant.schnet import SchNetCommittor
from committorch.models.equivariant.painn import PaiNNCommittor
from committorch.models.equivariant.graph_input import (
    build_graph,
    build_batch_graphs,
    scatter_mean,
    GraphInput,
)
from committorch.models.export import CommittorBiasForOpenMM
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

    def test_jacrev_lazy_encoder_forward(self):
        """``torch.func.jacrev`` on latent must not fail on `shape[-1]` (no-storage tensors)."""
        import torch.func as ffunc

        enc = DistanceEncoder(n_atoms=3, hidden_dim=16, output_dim=8)
        model = PretrainedBackbone(enc, feature_dim=8, freeze_encoder=False)
        x0 = torch.randn(3 * 3)

        def z_sum(x1d: torch.Tensor) -> torch.Tensor:
            return model.latent(x1d.unsqueeze(0)).sum()

        g = ffunc.jacrev(z_sum)(x0)
        assert g.shape == (9,)



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


class TestGraphInput:
    def test_build_graph_basic(self):
        positions = torch.randn(5, 3)
        atomic_numbers = torch.tensor([6, 6, 8, 1, 1])
        g = build_graph(positions, atomic_numbers, cutoff=10.0)

        assert isinstance(g, GraphInput)
        assert g.n_atoms == 5
        assert g.positions.shape == (5, 3)
        assert g.atomic_numbers.shape == (5,)
        assert g.edge_index.shape[0] == 2
        assert g.edge_vectors.shape[1] == 3
        assert g.edge_lengths.dim() == 1
        assert g.batch_index.shape == (5,)
        assert (g.batch_index == 0).all()

    def test_build_graph_no_self_loops(self):
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        atomic_numbers = torch.tensor([1, 1])
        g = build_graph(positions, atomic_numbers, cutoff=5.0, self_interaction=False)

        src, dst = g.edge_index
        assert not (src == dst).any()

    def test_build_graph_with_self_loops(self):
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        atomic_numbers = torch.tensor([1, 1])
        g = build_graph(positions, atomic_numbers, cutoff=5.0, self_interaction=True)

        n_edges = g.edge_index.shape[1]
        assert n_edges == 4  # 2 pairs + 2 self-loops

    def test_build_graph_cutoff_filtering(self):
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ])
        atomic_numbers = torch.tensor([1, 1, 1])
        g = build_graph(positions, atomic_numbers, cutoff=5.0)

        # Atom 2 is far from 0 and 1: only edges 0-1 and 1-0
        assert g.edge_index.shape[1] == 2

    def test_edge_vectors_correctness(self):
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
        ])
        atomic_numbers = torch.tensor([1, 1])
        g = build_graph(positions, atomic_numbers, cutoff=10.0)

        for i in range(g.edge_index.shape[1]):
            src, dst = g.edge_index[0, i], g.edge_index[1, i]
            expected = positions[dst] - positions[src]
            torch.testing.assert_close(
                g.edge_vectors[i], expected, atol=1e-6, rtol=1e-6
            )

        torch.testing.assert_close(
            g.edge_lengths[0],
            torch.tensor(5.0),
            atol=1e-5, rtol=1e-5,
        )

    def test_build_batch_graphs(self):
        positions = torch.randn(4, 3, 3)  # batch=4, n_atoms=3
        atomic_numbers = torch.tensor([6, 1, 1])
        g = build_batch_graphs(positions, atomic_numbers, cutoff=10.0)

        assert g.n_atoms == 12
        assert g.positions.shape == (12, 3)
        assert g.atomic_numbers.shape == (12,)
        assert g.batch_index.shape == (12,)

        # Each sample should have batch index 0,1,2,3
        for b in range(4):
            assert (g.batch_index[b*3:(b+1)*3] == b).all()


class TestScatterMean:
    def test_basic_pooling(self):
        src = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ])
        index = torch.tensor([0, 0, 1, 1])
        result = scatter_mean(src, index, dim_size=2)
        expected = torch.tensor([
            [2.0, 3.0],
            [6.0, 7.0],
        ])
        torch.testing.assert_close(result, expected)


class TestPlaceholderModels:
    """Test that placeholder MACE/SchNet/PaiNN models work without deps."""

    def test_mace_placeholder(self):
        model = MACECommittor(mace_model=None, feature_dim=32)
        x = torch.randn(5, 10)
        q = model(x)
        assert q.shape == (5,)

    def test_mace_placeholder_with_atomic_numbers(self):
        atomic_numbers = torch.tensor([6, 1, 1, 8, 1])
        model = MACECommittor(
            mace_model=None,
            atomic_numbers=atomic_numbers,
            feature_dim=32,
        )
        x = torch.randn(3, 15)
        q = model(x)
        assert q.shape == (3,)
        assert (q > 0).all() and (q < 1).all()

    def test_mace_latent(self):
        model = MACECommittor(mace_model=None, feature_dim=32)
        x = torch.randn(3, 10)
        z = model.latent(x)
        assert z.shape == (3,)

    def test_schnet_placeholder(self):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            model = SchNetCommittor(schnet_model=None, feature_dim=32)
        x = torch.randn(5, 10)
        q = model(x)
        assert q.shape == (5,)

    def test_painn_placeholder(self):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            model = PaiNNCommittor(painn_model=None, feature_dim=32)
        x = torch.randn(5, 10)
        q = model(x)
        assert q.shape == (5,)


class TestExportModule:
    def test_umbrella_bias_forward(self):
        from committorch.models.mlp import CommittorMLP
        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        export = CommittorBiasForOpenMM(
            model, bias_type="umbrella",
            bias_params={"target": 0.5, "spring_constant": 100.0},
            n_atoms=2,
        )
        pos = torch.randn(2, 3)
        e = export(pos)
        assert e.shape == ()
        assert e.item() >= 0

    def test_export_saves_file(self, tmp_path):
        from committorch.models.mlp import CommittorMLP
        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        export = CommittorBiasForOpenMM(
            model, bias_type="umbrella",
            bias_params={"target": 0.5, "spring_constant": 100.0},
            n_atoms=2,
        )
        out_path = export.export(tmp_path / "test_model.pt")
        assert out_path.exists()

    def test_exported_model_loads(self, tmp_path):
        from committorch.models.mlp import CommittorMLP
        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        export = CommittorBiasForOpenMM(
            model, bias_type="umbrella",
            bias_params={"target": 0.5, "spring_constant": 50.0},
            n_atoms=2,
        )
        out_path = export.export(tmp_path / "test_model.pt")
        loaded = torch.jit.load(str(out_path))
        pos = torch.randn(2, 3)
        e = loaded(pos)
        assert e.shape == ()

    def test_none_bias_returns_zero(self):
        from committorch.models.mlp import CommittorMLP
        model = CommittorMLP(input_dim=6, hidden_dims=[16])
        export = CommittorBiasForOpenMM(
            model, bias_type="none", n_atoms=2,
        )
        pos = torch.randn(2, 3)
        e = export(pos)
        assert e.item() == pytest.approx(0.0)


class TestPretrainedRegistry:
    def test_load_distance_encoder(self):
        model = load_pretrained("distance_encoder", n_atoms=5, freeze_encoder=True)
        pos = torch.randn(3, 5, 3)
        q = model(pos)
        assert q.shape == (3,)

    def test_load_mace_placeholder_direct(self):
        """MACECommittor with no model uses placeholder encoder."""
        model = MACECommittor(mace_model=None, feature_dim=32)
        x = torch.randn(3, 10)
        q = model(x)
        assert q.shape == (3,)

    def test_load_mace_requires_model_name_or_checkpoint(self):
        with pytest.raises(ValueError, match="model_name.*checkpoint_path"):
            load_pretrained("mace")

    def test_load_mace_with_checkpoint_no_atomic_numbers_raises(self):
        with pytest.raises(ValueError, match="atomic_numbers"):
            load_pretrained("mace", checkpoint_path="/fake/path.pt")

    def test_load_deprecated_schnet_warns(self):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            model = load_pretrained("schnet")
        x = torch.randn(3, 10)
        q = model(x)
        assert q.shape == (3,)

    def test_load_deprecated_painn_warns(self):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            model = load_pretrained("painn")
        x = torch.randn(3, 10)
        q = model(x)
        assert q.shape == (3,)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown model name"):
            load_pretrained("nonexistent_model")
