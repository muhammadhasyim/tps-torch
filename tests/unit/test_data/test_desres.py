"""Tests for DESRES data loader and brute-force rate computation."""

import torch
import pytest

from committorch.data.desres import (
    PROTEINS,
    compute_brute_force_rate,
    label_states,
)


class TestProteinRegistry:
    def test_expected_proteins(self):
        assert "CLN025" in PROTEINS
        assert "Trp-cage" in PROTEINS
        assert "BBA" in PROTEINS
        assert "villin" in PROTEINS

    def test_protein_metadata(self):
        cln = PROTEINS["CLN025"]
        assert cln.n_residues == 10
        assert cln.experimental_folding_time_us == pytest.approx(0.6)


class TestBruteForceRate:
    def _make_oscillating_rmsd(self, n_events: int = 5, frames_per_event: int = 100):
        """Create a synthetic RMSD trace with known folding/unfolding events."""
        rmsd_vals = []
        for i in range(n_events):
            rmsd_vals.extend([0.1] * frames_per_event)
            rmsd_vals.extend([0.8] * frames_per_event)
        rmsd_trace = torch.tensor(rmsd_vals, dtype=torch.float64)
        time = torch.arange(len(rmsd_vals), dtype=torch.float64)
        return rmsd_trace, time

    def test_counts_transitions(self):
        rmsd_trace, time = self._make_oscillating_rmsd(n_events=5)
        result = compute_brute_force_rate(
            rmsd_trace, time, folded_cutoff=0.2, unfolded_cutoff=0.6
        )
        assert result["n_folding_events"] >= 4
        assert result["n_unfolding_events"] >= 4

    def test_rate_is_positive(self):
        rmsd_trace, time = self._make_oscillating_rmsd(n_events=3)
        result = compute_brute_force_rate(
            rmsd_trace, time, folded_cutoff=0.2, unfolded_cutoff=0.6
        )
        assert result["folding_rate"] > 0
        assert result["unfolding_rate"] > 0

    def test_no_transitions(self):
        rmsd_trace = torch.ones(1000) * 0.1
        time = torch.arange(1000, dtype=torch.float64)
        result = compute_brute_force_rate(
            rmsd_trace, time, folded_cutoff=0.2, unfolded_cutoff=0.6
        )
        assert result["n_folding_events"] == 0
        assert result["folding_rate"] == 0.0

    def test_rate_scales_with_frequency(self):
        rmsd_fast, time_fast = self._make_oscillating_rmsd(n_events=10, frames_per_event=50)
        rmsd_slow, time_slow = self._make_oscillating_rmsd(n_events=2, frames_per_event=250)
        rate_fast = compute_brute_force_rate(rmsd_fast, time_fast)["folding_rate"]
        rate_slow = compute_brute_force_rate(rmsd_slow, time_slow)["folding_rate"]
        assert rate_fast > rate_slow


class TestLabelStates:
    def test_labels_folded_unfolded(self):
        rmsd_trace = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.1])
        labels = label_states(rmsd_trace, folded_cutoff=0.2, unfolded_cutoff=0.6)
        assert labels[0] == 1
        assert labels[1] == -1
        assert labels[2] == -1
        assert labels[3] == 0
        assert labels[4] == 1

    def test_all_folded(self):
        rmsd_trace = torch.ones(100) * 0.05
        labels = label_states(rmsd_trace, folded_cutoff=0.2)
        assert (labels == 1).all()
