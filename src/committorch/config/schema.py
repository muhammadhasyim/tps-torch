"""Pydantic configuration schemas for committorch runs.

All hyperparameters, system definitions, and method choices are captured
in typed, validated config objects rather than ad-hoc text files.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Method(str, Enum):
    """Enhanced sampling method selection."""

    BKE_UMBRELLA = "bke_umbrella"
    BKE_FTS = "bke_fts"
    BKE_VK_OPES = "bke_vk_opes"


class ModelConfig(BaseModel):
    """Configuration for the committor neural network."""

    architecture: str = "mlp"
    input_dim: int
    hidden_dims: list[int] = Field(default_factory=lambda: [200])
    activation: str = "ReLU"
    sigmoid_steepness: float = 1.0


class SimulatorConfig(BaseModel):
    """Configuration for the dynamics simulator."""

    potential: str = "muller_brown"
    kT: float = 1.0
    friction: float = 1.0
    dt: float = 1e-4
    dim: int = 2


class TrainingConfig(BaseModel):
    """Configuration for the training loop."""

    method: Method = Method.BKE_UMBRELLA
    n_iterations: int = 1000
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    lambda_boundary: float = 1.0
    lambda_supervised: float = 0.0
    n_simulation_steps: int = 100
    checkpoint_interval: int = 100


class RunConfig(BaseModel):
    """Top-level configuration combining all components."""

    model: ModelConfig
    simulator: SimulatorConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    seed: int = 42
