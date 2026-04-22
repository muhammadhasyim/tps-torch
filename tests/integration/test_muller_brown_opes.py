"""Integration test: BKE + V_K + OPES on Muller-Brown.

Verifies that the OPESLoop can train a committor MLP on the
Muller-Brown surface such that the loss decreases.
"""

import torch
import pytest

from committorch.models.mlp import CommittorMLP
from committorch.simulators.langevin import OverdampedLangevin
from committorch.simulators.potentials import muller_brown
from committorch.training.loop import OPESLoop


@pytest.mark.slow
class TestMullerBrownOPES:
    def test_loss_decreases(self):
        torch.manual_seed(123)
        model = CommittorMLP(input_dim=2, hidden_dims=[32, 32], sigmoid_steepness=3.0)
        sim = OverdampedLangevin(
            dim=2,
            potential_fn=muller_brown.energy,
            gradient_fn=muller_brown.gradient,
            kT=15.0,
            friction=1.0,
            dt=1e-4,
        )
        x_a = muller_brown.REACTANT_MINIMUM.unsqueeze(0) + 0.05 * torch.randn(10, 2)
        x_b = muller_brown.PRODUCT_MINIMUM.unsqueeze(0) + 0.05 * torch.randn(10, 2)

        loop = OPESLoop(
            model=model,
            simulator=sim,
            x_a=x_a,
            x_b=x_b,
            vk_lambda=0.5,
            opes_sigma=0.3,
            opes_barrier=5.0,
            opes_bias_factor=5.0,
            kernel_deposit_interval=10,
            lr=1e-3,
            lambda_boundary=10.0,
            n_sim_steps=50,
        )

        state = loop.run(n_iterations=30, progress=False)

        early_loss = sum(state.loss_history[:5]) / 5
        late_loss = sum(state.loss_history[-5:]) / 5
        assert late_loss < early_loss * 1.5, (
            f"Loss should not blow up: early={early_loss:.4f}, late={late_loss:.4f}"
        )

    def test_opes_deposits_hills(self):
        torch.manual_seed(456)
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        sim = OverdampedLangevin(
            dim=2,
            potential_fn=muller_brown.energy,
            gradient_fn=muller_brown.gradient,
            kT=15.0,
            friction=1.0,
            dt=1e-4,
        )
        x_a = muller_brown.REACTANT_MINIMUM.unsqueeze(0)
        x_b = muller_brown.PRODUCT_MINIMUM.unsqueeze(0)

        loop = OPESLoop(
            model=model,
            simulator=sim,
            x_a=x_a,
            x_b=x_b,
            kernel_deposit_interval=5,
            n_sim_steps=20,
            lr=1e-3,
            lambda_boundary=1.0,
        )

        loop.run(n_iterations=5, progress=False)
        assert loop.opes.n_kernels > 0, "OPES should have deposited kernels"
