# committorch

A PyTorch package for computing **committor functions** and **reaction rates** via ML-enhanced sampling.
Implements the methods described in:

1. M. R. Hasyim, C. H. Batton, K. K. Mandadapu,
   *"Supervised Learning and the Finite-Temperature String Method for Computing Committor Functions and Reaction Rates"*,
   [arXiv:2107.13522](https://arxiv.org/abs/2107.13522) (2021).

2. E. Trizio, S. Kang, M. Parrinello,
   *"Committor-Consistent Variational String Method"*,
   [arXiv:2410.17029](https://arxiv.org/abs/2410.17029) (2024).

## Overview

committorch trains neural-network committor functions using a self-consistent feedback loop between biased molecular dynamics and ML optimization. It supports:

- **BKE loss** (Backward Kolmogorov Equation) for committor training
- **Committor-based umbrella sampling** and the **Finite-Temperature String (FTS)** method (Paper 1)
- **Kolmogorov bias** (V_K) + **OPES metadynamics** for enhanced transition-state sampling (Paper 2, PLUMED-style algorithm)
- **Equivariant neural networks** via pre-trained MLIP backbones (MACE, SchNet, PaiNN)
- **OpenMM integration** through `openmm-torch` for atomistic simulations
- **Actor-learner distributed architecture** for scalable training
- **DESRES fast-folding protein benchmarks** (CLN025, Trp-cage, BBA, villin)
- Built-in toy potentials (Muller-Brown, 1D quartic) for validation

## Installation

```bash
pip install -e .
```

With optional dependencies:

```bash
pip install -e ".[all]"        # everything
pip install -e ".[openmm]"     # OpenMM + openmm-torch
pip install -e ".[equivariant]" # e3nn, MACE, SchNetPack
pip install -e ".[dev]"        # pytest, ruff, mypy
```

**Requirements:** Python >= 3.10, PyTorch >= 2.0

## Package Structure

```
src/committorch/
  models/           # Committor network architectures
    base.py         #   CommittorModel ABC
    mlp.py          #   Multi-layer perceptron
    equivariant/    #   MACE, SchNet, PaiNN committor heads
    pretrained.py   #   Pre-trained backbone registry
  simulators/       # Dynamics engines
    langevin.py     #   Overdamped Langevin integrator
    openmm_sim.py   #   OpenMM wrapper + openmm-torch modules
    potentials/     #   Muller-Brown, 1D quartic
  sampling/         # Enhanced sampling strategies
    umbrella.py     #   Committor-based umbrella sampling
    fts.py          #   Finite-Temperature String method
    kolmogorov.py   #   Kolmogorov bias V_K
    opes.py         #   OPES metadynamics (PLUMED-style)
  training/         # Training loops and losses
    losses.py       #   BKE, boundary, supervised, combined losses
    loop.py         #   Active learning loops (OPESLoop, UmbrellaLoop)
    reweighting.py  #   FEP and master-equation reweighting
    callbacks.py    #   Checkpointing, on-the-fly rate estimation
  analysis/         # Post-processing
    rates.py        #   Reaction rate computation
    fes.py          #   Free energy surfaces
    tse.py          #   Transition state ensemble extraction
    validation.py   #   Committor validation metrics
  distributed/      # Actor-learner parallelism
    comm.py         #   Communication backends
    learner.py      #   Central learner process
    worker.py       #   Simulation worker processes
  data/             # Data handling
    trajectory.py   #   Trajectory I/O (MDTraj)
    features.py     #   Structural features (distances, contacts, RMSD)
    datasets.py     #   PyTorch datasets
    desres.py       #   DESRES fast-folding protein benchmarks
  config/           # Pydantic configuration schemas
    schema.py
```

## Quick Start

```python
import torch
from committorch.models.mlp import CommittorMLP
from committorch.simulators.langevin import OverdampedLangevin
from committorch.simulators.potentials import muller_brown
from committorch.training.loop import OPESLoop

model = CommittorMLP(input_dim=2, hidden_dims=[64, 64])
sim = OverdampedLangevin(
    dim=2,
    potential_fn=muller_brown.energy,
    gradient_fn=muller_brown.gradient,
    kT=15.0, friction=1.0, dt=1e-4,
)

x_a = muller_brown.REACTANT_MINIMUM.unsqueeze(0) + 0.05 * torch.randn(20, 2)
x_b = muller_brown.PRODUCT_MINIMUM.unsqueeze(0) + 0.05 * torch.randn(20, 2)

loop = OPESLoop(
    model=model, simulator=sim, x_a=x_a, x_b=x_b,
    vk_lambda=1.0, opes_sigma=0.3, opes_barrier=5.0,
    lr=1e-3, lambda_boundary=10.0,
)
state = loop.run(n_iterations=100)
```

## Testing

```bash
pytest                          # all tests
pytest -m "not slow"            # skip integration tests
pytest tests/unit/              # unit tests only
```

## License

MIT
