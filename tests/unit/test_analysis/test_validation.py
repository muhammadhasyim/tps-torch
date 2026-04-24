"""Tests for protein-scale validation utilities."""

import torch
import pytest

from committorch.models.mlp import CommittorMLP
from committorch.analysis.validation import (
    mean_absolute_error,
    max_absolute_error,
    validate_committor_vs_brute_force,
    committor_rate_vs_brute_force,
)


class TestMeanAbsoluteError:
    def test_perfect_prediction(self):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        x = torch.randn(20, 2)
        with torch.no_grad():
            q_ref = model(x)
        mae = mean_absolute_error(model, x, q_ref)
        assert mae.item() == pytest.approx(0.0, abs=1e-6)


class TestMaxAbsoluteError:
    def test_perfect_prediction(self):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        x = torch.randn(20, 2)
        with torch.no_grad():
            q_ref = model(x)
        maxe = max_absolute_error(model, x, q_ref)
        assert maxe.item() == pytest.approx(0.0, abs=1e-6)


class TestValidateCommittorVsBruteForce:
    def test_perfect_agreement(self):
        """When committor perfectly matches RMSD labels, accuracy should be 1."""
        rmsd = torch.tensor([0.05, 0.05, 0.7, 0.7, 0.05])
        q_vals = torch.tensor([0.1, 0.2, 0.8, 0.9, 0.1])
        result = validate_committor_vs_brute_force(
            model=None,
            rmsd_trace=rmsd,
            committor_values=q_vals,
            folded_cutoff=0.2,
        )
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["n_frames"] == 5
        assert result["mean_q_folded"] < 0.5
        assert result["mean_q_unfolded"] > 0.5

    def test_random_disagreement(self):
        rmsd = torch.tensor([0.05, 0.05, 0.7, 0.7])
        q_vals = torch.tensor([0.9, 0.9, 0.1, 0.1])  # Inverted
        result = validate_committor_vs_brute_force(
            model=None,
            rmsd_trace=rmsd,
            committor_values=q_vals,
            folded_cutoff=0.2,
        )
        assert result["accuracy"] == pytest.approx(0.0)


class TestCommittorRateVsBruteForce:
    def _make_data(self, n_events=5, frames_per=100):
        rmsd_vals = []
        q_vals = []
        for _ in range(n_events):
            rmsd_vals.extend([0.1] * frames_per)
            q_vals.extend([0.1] * frames_per)
            rmsd_vals.extend([0.8] * frames_per)
            q_vals.extend([0.9] * frames_per)
        rmsd = torch.tensor(rmsd_vals, dtype=torch.float64)
        q = torch.tensor(q_vals, dtype=torch.float64)
        time = torch.arange(len(rmsd_vals), dtype=torch.float64)
        return q, rmsd, time

    def test_rates_are_positive(self):
        q, rmsd, time = self._make_data()
        result = committor_rate_vs_brute_force(
            q, rmsd, time,
            folded_cutoff=0.2,
            unfolded_cutoff=0.6,
        )
        assert result["brute_force_rate"] > 0
        assert result["committor_rate"] > 0
        assert result["brute_force_n_events"] >= 4

    def test_rate_ratio_close_to_one_for_consistent_data(self):
        q, rmsd, time = self._make_data(n_events=10)
        result = committor_rate_vs_brute_force(
            q, rmsd, time,
            folded_cutoff=0.2,
            unfolded_cutoff=0.6,
        )
        assert 0.3 < result["rate_ratio"] < 3.0
