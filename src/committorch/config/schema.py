"""Pydantic configuration schemas for committorch runs.

All hyperparameters, system definitions, and method choices are captured
in typed, validated config objects rather than ad-hoc text files.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Method(str, Enum):
    """Enhanced sampling method selection."""

    BKE_UMBRELLA = "bke_umbrella"
    BKE_FTS = "bke_fts"
    BKE_VK_OPES = "bke_vk_opes"


class Architecture(str, Enum):
    """Committor model architecture."""

    MLP = "mlp"
    MACE = "mace"
    SCHNET = "schnet"
    PAINN = "painn"
    DISTANCE_ENCODER = "distance_encoder"


class SimulatorEngine(str, Enum):
    """Dynamics engine."""

    OVERDAMPED_LANGEVIN = "overdamped_langevin"
    OPENMM = "openmm"


class ModelConfig(BaseModel):
    """Configuration for the committor neural network."""

    architecture: Architecture = Architecture.MLP
    input_dim: int = 2
    hidden_dims: list[int] = Field(default_factory=lambda: [200])
    activation: str = "ReLU"
    sigmoid_steepness: float = 1.0

    # GNN-specific
    checkpoint_path: Optional[str] = None
    cutoff: float = 5.0
    feature_dim: int = 128
    head_hidden_dims: list[int] = Field(default_factory=lambda: [32, 32])
    freeze_encoder: bool = True
    n_atoms: int = 0


class SimulatorConfig(BaseModel):
    """Configuration for the dynamics simulator."""

    engine: SimulatorEngine = SimulatorEngine.OVERDAMPED_LANGEVIN
    potential: str = "muller_brown"
    kT: float = 1.0
    friction: float = 1.0
    dt: float = 1e-4
    dim: int = 2

    # OpenMM-specific
    platform: str = "CPU"
    precision: str = "mixed"


class ProteinConfig(BaseModel):
    """Configuration for protein folding simulations."""

    pdb_path: str = ""
    forcefield: list[str] = Field(
        default_factory=lambda: ["amber14-all.xml", "amber14/tip3pfb.xml"]
    )
    temperature_K: float = 340.0
    pressure_atm: float = 1.0
    timestep_fs: float = 2.0
    solvent: str = "explicit"
    ionic_strength_M: float = 0.0
    nonbonded_cutoff_nm: float = 1.0
    hydrogen_mass_amu: float = 1.5
    padding_nm: float = 1.0


class OPESConfig(BaseModel):
    """OPES metadynamics hyperparameters."""

    sigma: float = 0.5
    barrier: float = 5.0
    bias_factor: float = 10.0
    compression_threshold: float = 1.0
    kernel_deposit_interval: int = 100


class UmbrellaConfig(BaseModel):
    """Umbrella sampling hyperparameters."""

    n_windows: int = 10
    spring_constant: float = 100.0


class KolmogorovConfig(BaseModel):
    """Kolmogorov bias hyperparameters."""

    lam: float = 1.0


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
    collect_interval: int = 50
    checkpoint_interval: int = 100

    opes: OPESConfig = Field(default_factory=OPESConfig)
    umbrella: UmbrellaConfig = Field(default_factory=UmbrellaConfig)
    kolmogorov: KolmogorovConfig = Field(default_factory=KolmogorovConfig)


class RunConfig(BaseModel):
    """Top-level configuration combining all components."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    simulator: SimulatorConfig = Field(default_factory=SimulatorConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    protein: Optional[ProteinConfig] = None
    seed: int = 42
    output_dir: str = "output"
    log_level: str = "INFO"
