"""Shared fixtures for committorch test suite."""

import pytest
import torch


@pytest.fixture
def default_dtype():
    """Ensure float64 for scientific accuracy during tests."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield torch.float64
    torch.set_default_dtype(old)


@pytest.fixture
def rng():
    """Seeded random generator for reproducibility."""
    return torch.Generator().manual_seed(42)
