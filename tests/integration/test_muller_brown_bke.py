"""Integration test: BKE + umbrella sampling on Muller-Brown.

Verifies that the UmbrellaLoop can train a committor MLP on the
Muller-Brown surface such that:
- q(reactant) is near 0
- q(product) is near 1
- The loss decreases over training
"""

import torch
import pytest

from committorch.models.mlp import CommittorMLP
from committorch.simulators.langevin import OverdampedLangevin
from committorch.simulators.potentials import muller_brown
from committorch.training.loop import UmbrellaLoop


@pytest.mark.slow
class TestMullerBrownUmbrella:
    def test_loss_decreases(self):
        torch.manual_seed(0)
        model = CommittorMLP(input_dim=2, hidden_dims=[32, 32])
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

        loop = UmbrellaLoop(
            model=model,
            simulator=sim,
            x_a=x_a,
            x_b=x_b,
            n_windows=5,
            spring_constant=50.0,
            lr=1e-3,
            lambda_boundary=10.0,
            n_sim_steps=50,
        )

        state = loop.run(n_iterations=50, progress=False)

        early_loss = sum(state.loss_history[:5]) / 5
        late_loss = sum(state.loss_history[-5:]) / 5
        assert late_loss < early_loss, (
            f"Loss should decrease: early={early_loss:.4f}, late={late_loss:.4f}"
        )

    def test_boundary_conditions(self):
        torch.manual_seed(0)
        model = CommittorMLP(input_dim=2, hidden_dims=[32, 32])
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

        loop = UmbrellaLoop(
            model=model,
            simulator=sim,
            x_a=x_a,
            x_b=x_b,
            n_windows=5,
            spring_constant=50.0,
            lr=1e-3,
            lambda_boundary=10.0,
            n_sim_steps=50,
        )

        loop.run(n_iterations=100, progress=False)

        with torch.no_grad():
            q_a = model(muller_brown.REACTANT_MINIMUM.unsqueeze(0))
            q_b = model(muller_brown.PRODUCT_MINIMUM.unsqueeze(0))

        assert q_a.item() < 0.4, f"q(reactant)={q_a.item():.3f} should be near 0"
        assert q_b.item() > 0.6, f"q(product)={q_b.item():.3f} should be near 1"
