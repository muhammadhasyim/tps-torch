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
- **OpenMM integration** through **openmmtorch** (PyPI) / `openmm-torch` (conda) for atomistic simulations
- **Actor-learner distributed architecture** for scalable training
- **DESRES fast-folding protein benchmarks** (CLN025, Trp-cage, BBA, villin)
- Built-in toy potentials (Muller-Brown, 1D quartic) for validation

## Installation

**Recommended (Conda):** use the checked-in `environment.yml` so **conda-forge** supplies a consistent stack: OpenMM, `libstdc++` (avoids `CXXABI_*` errors when the system `libstdc++` is too old), `openmm-torch` (PyTorch OpenMM plugin; same role as the PyPI package `openmmtorch`), PyTorch, PDBFixer, mdtraj, and dev tools. The only **pip** step is an editable install of this package and its **dev** extras.

```bash
cd /path/to/tps-torch   # repository root — required for the editable install in the YAML
conda env create -f environment.yml
conda activate tpstorch
```

That runs `pip install -e ".[dev]"` after the Conda dependencies resolve (so no second OpenMM or mixed pip/conda C++ runtimes for the core stack).

**Linux — `CXXABI_*` / `libstdc++.so.6` errors when importing OpenMM:** the loader was using the **system** C++ runtime instead of Conda’s. This repository calls `committorch.conda_ld_path.ensure_conda_lib_path()` before loading OpenMM/torch in CLN025 scripts and in `OpenMMSimulator`, which prepends `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH` in the current process. After `conda activate`, you can also run `source scripts/conda-ld-library-path.sh` in each new shell. If you still see the error, your `tpstorch` env may predate `environment.yml` (e.g. Python 3.14 from an ad-hoc install): remove the env and recreate with the steps in the header of `environment.yml`.

**Linux — OpenMM CUDA and `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` (222):** this is a **driver / CUDA user toolkit** mismatch, not a bug in the CLN025 scripts. The `environment.yml` pins **`openmm` to 8.2.x** (with `openmm-torch` 1.5) because conda-forge’s **OpenMM 8.5** builds are aimed at **CUDA 12.9+ / 13**; a typical `nvidia-smi` “CUDA Version” of **12.4 or 12.6** is often **too old** to JIT the PTX that ships in those 8.5 binaries, which triggers 222.

- **A — Upgrade the NVIDIA driver** to a production release whose supported CUDA is **≥ 12.9** (check the [driver / CUDA table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-toolkit-driver-version)). Then you can `conda install -c conda-forge "openmm=8.5"` if you need the latest OpenMM.
- **B — Stay on the current driver** and use the **8.2.x + CUDA 11.8** user stack: recreate the env with **`environment-cuda-linux.yml`**, or in an active env run  
  `conda install -c conda-forge cudatoolkit=11.8 "openmm>=8.2,<8.3"`  
  and reinstall so Conda’s `libstdc++` and `cudatoolkit` stay aligned.
- **C — Diagnose quickly:** from the repo root, run `python scripts/verify_openmm_cuda.py` (tiny CUDA `Context` test and driver summary).

**Optional:** NVIDIA GPU PyTorch — after the CPU env works, install a CUDA build from the same channel, for example:

```bash
conda activate tpstorch
conda install -c conda-forge "pytorch=*=*cuda*"  # adjust CUDA level to match your driver; remove pytorch-cpu if conda asks
```

**Pure pip** (no conda):

```bash
pip install -e .
```

With optional dependencies:

```bash
pip install -e ".[all]"        # everything
pip install -e ".[openmm]"     # OpenMM + openmmtorch from PyPI (may mix with system libstdc++)
pip install -e ".[openmm-from-conda]"  # only openmmtorch — use when OpenMM is from conda
pip install -e ".[equivariant]" # e3nn, MACE, SchNetPack
pip install -e ".[dev]"        # pytest, ruff, mypy
```

**Requirements:** Python >= 3.10, PyTorch >= 2.0, `pip` >= 22 (for editable installs; run `python -m pip install -U pip` if you see a missing `build_editable` error).

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
