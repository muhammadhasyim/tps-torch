"""Integration test: download MACE-OFF-small and run a forward pass.

This test downloads the MACE-OFF23 small model (~32 MB) on first run
(cached in ``~/.cache/mace/``), builds a ``MACECommittor``, and verifies
end-to-end forward/latent/gradient computation on a 22-atom alanine
dipeptide structure.

Requires: ``mace-torch`` and ``e3nn`` (``pip install mace-torch``).
"""

import os

import pytest
import torch

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

try:
    from committorch.models.equivariant.mace import (
        HAS_MACE,
        MACECommittor,
        download_mace_model,
        parse_foundation_model_spec,
    )
    from committorch.models.pretrained import load_pretrained
except ImportError:
    HAS_MACE = False

# Literals in angstroms (for readability); committor/MACE use nm like OpenMM.
ALANINE_DIPEPTIDE_POSITIONS = (
    torch.tensor(
        [
            [2.0000, 1.0000, 0.0000],
            [2.0000, 2.0900, 0.0000],
            [3.0900, 2.0900, 0.0000],
            [3.0900, 3.1800, 0.0000],
            [4.1800, 2.0900, 0.0000],
            [4.1800, 3.1800, 0.0000],
            [5.2700, 2.0900, 0.0000],
            [5.2700, 1.0000, 0.0000],
            [6.3600, 2.0900, 0.0000],
            [6.3600, 3.1800, 0.0000],
            [7.4500, 2.0900, 0.0000],
            [7.4500, 1.0000, 0.0000],
            [1.4850, 0.6550, -0.8900],
            [1.4850, 0.6550, 0.8900],
            [1.0000, 1.0000, 0.0000],
            [1.5100, 2.4350, -0.8900],
            [1.5100, 2.4350, 0.8900],
            [4.1800, 4.0900, 0.0000],
            [5.1650, 3.5250, 0.0000],
            [6.3600, 4.0900, 0.0000],
            [8.4400, 2.0900, 0.0000],
            [7.4500, 0.3450, 0.0000],
        ],
        dtype=torch.float64,
    )
    * 0.1
)

ALANINE_DIPEPTIDE_ATOMIC_NUMBERS = torch.tensor(
    [6, 6, 6, 8, 7, 6, 6, 8, 7, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    dtype=torch.long,
)


@pytest.mark.equivariant
@pytest.mark.slow
class TestMACEFoundationModel:
    """Tests requiring a real MACE model download (~32 MB, cached)."""

    @pytest.fixture(scope="class")
    def mace_model(self):
        if not HAS_MACE:
            pytest.skip("mace-torch not installed")
        return download_mace_model("mace-off", "small", device="cpu")

    def test_download_mace_off_small(self, mace_model):
        assert isinstance(mace_model, torch.nn.Module)
        n_params = sum(p.numel() for p in mace_model.parameters())
        assert n_params > 100_000

    def test_mace_has_r_max(self, mace_model):
        assert hasattr(mace_model, "r_max")
        assert float(mace_model.r_max) > 0

    def test_mace_committor_forward(self, mace_model):
        model = MACECommittor(
            mace_model=mace_model,
            atomic_numbers=ALANINE_DIPEPTIDE_ATOMIC_NUMBERS,
            feature_dim=64,
            head_hidden_dims=(32,),
            freeze_encoder=True,
        )
        pos = ALANINE_DIPEPTIDE_POSITIONS.unsqueeze(0).float()
        q = model(pos)
        assert q.shape == (1,)
        assert 0.0 < q.item() < 1.0

    def test_mace_committor_latent(self, mace_model):
        model = MACECommittor(
            mace_model=mace_model,
            atomic_numbers=ALANINE_DIPEPTIDE_ATOMIC_NUMBERS,
            feature_dim=64,
            head_hidden_dims=(32,),
            freeze_encoder=True,
        )
        pos = ALANINE_DIPEPTIDE_POSITIONS.unsqueeze(0).float()
        z = model.latent(pos)
        assert z.shape == (1,)
        assert torch.isfinite(z).all()

    def test_mace_committor_flat_input(self, mace_model):
        """Flat (batch, n_atoms*3) input should work."""
        model = MACECommittor(
            mace_model=mace_model,
            atomic_numbers=ALANINE_DIPEPTIDE_ATOMIC_NUMBERS,
            feature_dim=64,
            head_hidden_dims=(32,),
            freeze_encoder=True,
        )
        pos_flat = ALANINE_DIPEPTIDE_POSITIONS.reshape(1, -1).float()
        q = model(pos_flat)
        assert q.shape == (1,)

    def test_mace_committor_batch(self, mace_model):
        model = MACECommittor(
            mace_model=mace_model,
            atomic_numbers=ALANINE_DIPEPTIDE_ATOMIC_NUMBERS,
            feature_dim=64,
            head_hidden_dims=(32,),
            freeze_encoder=True,
        )
        pos = ALANINE_DIPEPTIDE_POSITIONS.unsqueeze(0).expand(3, -1, -1).float()
        pos = pos + 0.01 * torch.randn_like(pos)
        q = model(pos)
        assert q.shape == (3,)

    def test_from_foundation_method(self):
        if not HAS_MACE:
            pytest.skip("mace-torch not installed")
        model = MACECommittor.from_foundation(
            model_name="mace-off",
            size="small",
            atomic_numbers=ALANINE_DIPEPTIDE_ATOMIC_NUMBERS,
            feature_dim=64,
        )
        pos = ALANINE_DIPEPTIDE_POSITIONS.unsqueeze(0).float()
        q = model(pos)
        assert q.shape == (1,)

    def test_from_pretrained_with_model_name(self):
        if not HAS_MACE:
            pytest.skip("mace-torch not installed")
        model = MACECommittor.from_pretrained(
            model_name="mace-off",
            size="small",
            atomic_numbers=ALANINE_DIPEPTIDE_ATOMIC_NUMBERS,
            feature_dim=64,
        )
        pos = ALANINE_DIPEPTIDE_POSITIONS.unsqueeze(0).float()
        q = model(pos)
        assert q.shape == (1,)

    def test_load_pretrained_shorthand(self):
        if not HAS_MACE:
            pytest.skip("mace-torch not installed")
        model = load_pretrained(
            "mace-off-small",
            atomic_numbers=ALANINE_DIPEPTIDE_ATOMIC_NUMBERS,
            feature_dim=64,
        )
        pos = ALANINE_DIPEPTIDE_POSITIONS.unsqueeze(0).float()
        q = model(pos)
        assert q.shape == (1,)

    def test_encoder_frozen_by_default(self, mace_model):
        model = MACECommittor(
            mace_model=mace_model,
            atomic_numbers=ALANINE_DIPEPTIDE_ATOMIC_NUMBERS,
            feature_dim=64,
            freeze_encoder=True,
        )
        for p in model.encoder.mace.parameters():
            assert not p.requires_grad
        for p in model.head.parameters():
            assert p.requires_grad

    def test_gradient_through_head(self, mace_model):
        """model.gradient() should return non-zero gradients."""
        model = MACECommittor(
            mace_model=mace_model,
            atomic_numbers=ALANINE_DIPEPTIDE_ATOMIC_NUMBERS,
            feature_dim=64,
            head_hidden_dims=(32,),
            freeze_encoder=True,
        )
        pos = ALANINE_DIPEPTIDE_POSITIONS.unsqueeze(0).float()
        grad = model.gradient(pos)
        assert grad.shape == pos.shape
        assert grad.abs().sum() > 0


@pytest.mark.equivariant
class TestParseFoundationModelSpec:
    def test_parse_mace_off_small(self):
        fam, sz = parse_foundation_model_spec("mace-off-small")
        assert fam == "mace-off"
        assert sz == "small"

    def test_parse_mace_mp_large(self):
        fam, sz = parse_foundation_model_spec("mace-mp-large")
        assert fam == "mace-mp"
        assert sz == "large"

    def test_parse_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_foundation_model_spec("unknown-model-small")
