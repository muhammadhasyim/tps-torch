"""Train a committor function for CLN025 protein folding.

This script implements the full feedback loop:
1. Load or build the OpenMM system from setup_system.py output
2. Initialize a committor model (MACE or DistanceEncoder)
3. Generate initial boundary configurations from folded/unfolded PDB
4. Run the active learning loop (ProteinOPESLoop or ProteinUmbrellaLoop)
5. Periodically checkpoint model + OPES state

Usage
-----
    With defaults (paths relative to this example directory):
    python train_committor.py --system_xml output/system.xml \\
                              --pdb_folded output/minimized.pdb \\
                              --pdb_unfolded output/unfolded_solvated.pdb

``system.xml`` is built from a **solvated** box; ``pdb_folded`` / ``pdb_unfolded`` must
be **full solvated** PDBs with the same atom count as that system (e.g. from
``setup_system.py`` and ``make_unfolded_snapshot.py``), not crystal-only RCSB files.
Optionally set ``--reference_pdb`` to the topology file if it differs from
``--pdb_folded`` (both must still match ``system.xml``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from conda_ensure import ensure_conda_lib_path_early

ensure_conda_lib_path_early()

import torch

logger = logging.getLogger(__name__)

_OPENMM_LOAD_ERROR: Exception | None = None
try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    HAS_OPENMM = True
except Exception as exc:  # ImportError, OSError (bad .so), etc.
    HAS_OPENMM = False
    _OPENMM_LOAD_ERROR = exc


def _openmm_missing_message() -> str:
    """Explain why OpenMM did not load (wrong env, missing package, bad library)."""
    base = (
        "OpenMM could not be loaded in this Python interpreter.\n"
        f"  Executable: {sys.executable}\n"
        "Install OpenMM **in the same environment** you use to run this script, e.g.:\n"
        "  conda install -n tpstorch -c conda-forge openmm\n"
        "  # or:  python -m pip install openmm\n"
    )
    if _OPENMM_LOAD_ERROR is not None:
        base += f"  Underlying error: {_OPENMM_LOAD_ERROR!r}\n"
        err = f"{_OPENMM_LOAD_ERROR}"
        if "CXXABI" in err or "libstdc++" in err:
            base += (
                "  (Linux) Conda’s libstdc++ must be found before the system’s: "
                "this script prepends $CONDA_PREFIX/lib; ensure you "
                "‘conda activate’ the env. Recreate the env with the root "
                "environment.yml (python=3.12, libstdcxx-ng, openmm from conda-forge).\n"
            )
    return base


def _assert_pdb_matches_system(
    system: "openmm.System", topology: "app.Topology", label: str
) -> None:
    """Require PDB topology atom count to match the serialized System particle count."""
    n_sys = system.getNumParticles()
    n_pdb = topology.getNumAtoms()
    if n_sys != n_pdb:
        raise ValueError(
            f"{label}: PDB has {n_pdb} atoms but system.xml has {n_sys} particles. "
            "Use solvated PDBs from the same run as system.xml (e.g. output/minimized.pdb "
            "from setup_system.py and output/unfolded_solvated.pdb from "
            "make_unfolded_snapshot.py). A crystal-only protein PDB will not match."
        )


def _require_input_paths(
    system_xml: str,
    pdb_folded: str,
    pdb_unfolded: str,
    reference_pdb: str | None,
) -> None:
    """Fail fast with a clear message if inputs are missing."""
    items: list[tuple[str, str]] = [
        ("--system_xml", system_xml),
        ("--pdb_folded", pdb_folded),
        ("--pdb_unfolded", pdb_unfolded),
    ]
    if reference_pdb:
        items.append(("--reference_pdb", reference_pdb))
    missing = [f"  {flag} {path!r}" for flag, path in items if not Path(path).is_file()]
    if not missing:
        return
    msg = "Required file(s) not found (paths are relative to your current working directory):\n"
    msg += "\n".join(missing)
    if not Path(pdb_unfolded).is_file():
        msg += (
            "\nTo create the unfolded solvated structure, run from this example directory:\n"
            "  python make_unfolded_snapshot.py --input_pdb output/minimized.pdb \\\n"
            "      --system_xml output/system.xml --output output/unfolded_solvated.pdb"
        )
    raise FileNotFoundError(msg)


def _remove_barostat_for_nvt(simulation_system: "openmm.System") -> None:
    """Drop MonteCarlo barostat from ``setup_system`` output.

    ``system.xml`` is often built with NPT; Langevin + 2 fs + new velocities on that
    system commonly hits NaN. Committor boundary sampling and short MD here use NVT.
    """
    n = simulation_system.getNumForces()
    removed = 0
    for i in range(n - 1, -1, -1):
        if isinstance(simulation_system.getForce(i), openmm.MonteCarloBarostat):
            simulation_system.removeForce(i)
            removed += 1
    if removed:
        logger.info(
            "Removed %d MonteCarloBarostat force(s); using NVT for training/boundary MD.",
            removed,
        )


def load_boundary_configs(
    pdb_path: str,
    n_configs: int = 10,
    n_equilibrate_steps: int = 1000,
    simulation: Any = None,
) -> torch.Tensor:
    """Generate boundary configurations from a PDB structure.

    Runs short unbiased MD from the PDB starting structure
    to collect a small ensemble of configurations near that basin.

    Parameters
    ----------
    pdb_path : str
        Path to PDB file (folded or unfolded).
    n_configs : int
        Number of configs to collect.
    n_equilibrate_steps : int
        MD steps between config snapshots.
    simulation : openmm.app.Simulation or None
        If provided, set positions from PDB and run MD.

    Returns
    -------
    torch.Tensor, shape (n_configs, n_atoms * 3)
    """
    if simulation is None:
        raise RuntimeError("OpenMM simulation required for boundary configs")

    pdb = app.PDBFile(pdb_path)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy(maxIterations=500)
    simulation.context.setVelocitiesToTemperature(340.0 * unit.kelvin)

    configs = []
    for _ in range(n_configs):
        simulation.step(n_equilibrate_steps)
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        import numpy as np
        pos_nm = np.array(pos.value_in_unit(unit.nanometers))
        configs.append(torch.tensor(pos_nm.reshape(1, -1), dtype=torch.float32))

    return torch.cat(configs, dim=0)


def build_model(
    foundation_model: str = "mace-off-small",
    n_atoms: int = 10,
    atomic_numbers: torch.Tensor | None = None,
    checkpoint_path: str | None = None,
    feature_dim: int = 128,
    freeze_encoder: bool = True,
    hidden_dim: int = 8,
    device: str = "cpu",
) -> torch.nn.Module:
    """Build the committor model.

    Parameters
    ----------
    foundation_model : str
        Model specifier.  Foundation models (recommended):
        ``"mace-off-small"``, ``"mace-off-medium"``, ``"mace-mp-small"``, etc.
        Test-only: ``"distance_encoder"``.
    n_atoms : int
        Number of atoms (used only for ``distance_encoder``).
    atomic_numbers : torch.Tensor or None
        Atomic numbers for the system (required for MACE models).
    checkpoint_path : str or None
        Path to custom MACE checkpoint (overrides auto-download).
    feature_dim : int
        Encoder output dimension.
    freeze_encoder : bool
        Whether to freeze the encoder.
    hidden_dim : int
        Hidden dim for distance_encoder MLP (test-only).
    device : str
        Target device for the model.

    Returns
    -------
    CommittorModel
    """
    from committorch.models.pretrained import load_pretrained

    if foundation_model == "distance_encoder":
        return load_pretrained(
            "distance_encoder",
            n_atoms=n_atoms,
            hidden_dim=hidden_dim,
            freeze_encoder=freeze_encoder,
        )

    if checkpoint_path is not None:
        return load_pretrained(
            "mace",
            checkpoint_path=checkpoint_path,
            atomic_numbers=atomic_numbers,
            feature_dim=feature_dim,
            freeze_encoder=freeze_encoder,
            device=device,
        )

    return load_pretrained(
        foundation_model,
        atomic_numbers=atomic_numbers,
        feature_dim=feature_dim,
        freeze_encoder=freeze_encoder,
        device=device,
    )


def _get_atomic_numbers_from_topology(topology) -> torch.Tensor:
    """Extract atomic numbers from an OpenMM Topology."""
    from openmm.app import Element

    nums = []
    for atom in topology.atoms():
        el = atom.element
        if el is not None:
            nums.append(el.atomic_number)
        else:
            nums.append(0)
    return torch.tensor(nums, dtype=torch.long)


def train(
    system_xml: str,
    pdb_folded: str,
    pdb_unfolded: str,
    reference_pdb: str | None = None,
    foundation_model: str = "mace-off-small",
    checkpoint_path: str | None = None,
    method: str = "opes",
    n_iterations: int = 100,
    n_sim_steps: int = 1000,
    collect_interval: int = 100,
    kernel_deposit_interval: int = 50,
    lr: float = 1e-3,
    checkpoint_interval: int = 10,
    output_dir: str = "output",
    platform: str = "CPU",
    temperature_K: float = 340.0,
    vk_lambda: float = 1.0,
    opes_barrier: float = 5.0,
    opes_sigma: float = 0.5,
    n_windows: int = 10,
    spring_constant: float = 100.0,
    bias_type: str = "umbrella",
    hidden_dim: int = 8,
    require_cuda: bool = False,
    strict_openmm_platform: bool = False,
    feature_dim: int = 128,
    resume_from: str | None = None,
) -> None:
    """Run the committor training loop for CLN025.

    Parameters
    ----------
    system_xml : str
        Path to serialized OpenMM System XML.
    pdb_folded : str
        Path to folded-state coordinates (same atom count as ``system.xml``).
    pdb_unfolded : str
        Path to unfolded-state coordinates (same atom count as ``system.xml``).
    reference_pdb : str or None
        Topology used to build ``Simulation`` (must match ``system.xml``).
    foundation_model : str
        Foundation model specifier: ``"mace-off-small"`` (default),
        ``"mace-off-medium"``, ``"mace-mp-small"``, etc. for auto-download
        of a pre-trained MACE backbone.  ``"distance_encoder"`` for the
        test-only pairwise distance MLP.
    checkpoint_path : str or None
        Custom MACE checkpoint path (overrides auto-download).
    method : str
        ``"opes"`` for V_K + OPES, ``"umbrella"`` for umbrella windows.
    n_iterations : int
        Number of outer loop iterations.
    n_sim_steps : int
        MD steps per collection phase.
    collect_interval : int
        Collect snapshot every N steps.
    kernel_deposit_interval : int
        Deposit OPES kernel every N steps.
    lr : float
        Learning rate.
    checkpoint_interval : int
        Save checkpoint every N iterations.
    output_dir : str
        Output directory for checkpoints and logs.
    platform : str
        OpenMM platform.
    temperature_K : float
        Temperature in Kelvin.
    vk_lambda : float
        Kolmogorov bias strength.
    opes_barrier : float
        OPES barrier height in kT.
    opes_sigma : float
        OPES kernel width.
    n_windows : int
        Number of umbrella windows (for umbrella method).
    spring_constant : float
        Umbrella spring constant.
    hidden_dim : int
        Hidden dim for distance_encoder MLP (test-only).
    require_cuda : bool
        If True, require PyTorch CUDA and ``--platform CUDA`` (no CPU training).
    strict_openmm_platform : bool
        If True, OpenMM does not fall back to OpenCL/CPU if CUDA fails.
    feature_dim : int
        MACE (and distance_encoder output) feature dimension.
    resume_from : str or None
        Path to a checkpoint file produced by a previous run.  Restores
        model weights, optimizer state, and iteration counter so training
        continues from where it left off.
    """
    if not HAS_OPENMM:
        raise ImportError(_openmm_missing_message()) from _OPENMM_LOAD_ERROR

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "--require_cuda was set but torch.cuda.is_available() is False. "
            "Install a CUDA build of PyTorch and ensure a GPU is visible."
        )
    if require_cuda and platform.strip().upper() != "CUDA":
        raise ValueError(
            f"--require_cuda requires --platform CUDA (got {platform!r})"
        )

    from committorch.openmm_platforms import create_simulation_with_platform_fallback

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _require_input_paths(system_xml, pdb_folded, pdb_unfolded, reference_pdb)

    # Load system
    logger.info("Loading system from %s", system_xml)
    with open(system_xml) as f:
        system = openmm.XmlSerializer.deserialize(f.read())
    _remove_barostat_for_nvt(system)

    ref_path = reference_pdb or pdb_folded
    ref = app.PDBFile(ref_path)
    _assert_pdb_matches_system(system, ref.topology, f"reference PDB ({ref_path})")
    fold = app.PDBFile(pdb_folded)
    _assert_pdb_matches_system(system, fold.topology, f"pdb_folded ({pdb_folded})")
    _assert_pdb_matches_system(
        system,
        app.PDBFile(pdb_unfolded).topology,
        f"pdb_unfolded ({pdb_unfolded})",
    )

    temperature = temperature_K * unit.kelvin
    # 1 fs is much more stable than 2 fs for solvated systems after PDB reload + Langevin
    integrator = openmm.LangevinMiddleIntegrator(
        temperature, 1.0 / unit.picosecond, 1.0 * unit.femtoseconds
    )
    simulation = create_simulation_with_platform_fallback(
        ref.topology,
        system,
        integrator,
        platform,
        log=logger,
        strict=strict_openmm_platform,
    )
    simulation.context.setPositions(fold.positions)
    n_atoms = simulation.topology.getNumAtoms()

    kT = (unit.MOLAR_GAS_CONSTANT_R * temperature).value_in_unit(
        unit.kilojoules_per_mole
    )

    # Extract atomic numbers from topology for MACE models
    atomic_numbers = _get_atomic_numbers_from_topology(simulation.topology)

    # Build model
    logger.info("Building %s model with %d atoms", foundation_model, n_atoms)
    if require_cuda:
        torch_device_str = "cuda"
    else:
        torch_device_str = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(
        foundation_model=foundation_model,
        n_atoms=n_atoms,
        atomic_numbers=atomic_numbers,
        checkpoint_path=checkpoint_path,
        hidden_dim=hidden_dim,
        feature_dim=feature_dim,
        device=torch_device_str,
    )

    device = torch.device(torch_device_str)
    model = model.to(device)
    logger.info("Model device: %s", device)

    # Generate boundary configs
    logger.info("Generating folded boundary configs...")
    x_a = load_boundary_configs(pdb_folded, simulation=simulation)
    logger.info("Generating unfolded boundary configs...")
    x_b = load_boundary_configs(pdb_unfolded, simulation=simulation)

    # Boundary configs must be on the same device as the model
    x_a = x_a.to(device)
    x_b = x_b.to(device)

    # Create simulator
    from committorch.simulators.openmm_sim import OpenMMSimulator
    sim = OpenMMSimulator(
        simulation=simulation,
        model=model,
        kT=kT,
    )

    # Create training loop
    if method == "opes":
        from committorch.training.protein_loop import ProteinOPESLoop
        loop = ProteinOPESLoop(
            model=model,
            simulator=sim,
            x_a=x_a,
            x_b=x_b,
            vk_lambda=vk_lambda,
            opes_sigma=opes_sigma,
            opes_barrier=opes_barrier,
            kernel_deposit_interval=kernel_deposit_interval,
            collect_interval=collect_interval,
            n_sim_steps=n_sim_steps,
            lr=lr,
            bias_type=bias_type,
        )
    elif method == "umbrella":
        from committorch.training.protein_loop import ProteinUmbrellaLoop
        loop = ProteinUmbrellaLoop(
            model=model,
            simulator=sim,
            x_a=x_a,
            x_b=x_b,
            n_windows=n_windows,
            spring_constant=spring_constant,
            n_sim_steps=n_sim_steps,
            lr=lr,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Resume from checkpoint if requested
    start_iter = 0
    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
        logger.info("Resuming from checkpoint %s", resume_from)
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        loop.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iter = ckpt["iteration"]
        loop.state.iteration = start_iter
        loop.state.loss_history = ckpt.get("loss_history", [])
        loop.state.bke_history = ckpt.get("bke_history", [])
        loop.state.boundary_history = ckpt.get("boundary_history", [])
        logger.info("Resumed at iteration %d", start_iter)

    # Training loop with checkpointing
    logger.info("Starting training: %d iterations (from %d)", n_iterations, start_iter)
    for it in range(start_iter, n_iterations):
        configs, weights = loop.collect_samples()
        loss = loop.train_step(configs, weights)
        loop.update_bias()
        loop.state.iteration += 1

        logger.info(
            "Iteration %d: loss=%.4e, bke=%.4e, boundary=%.4e",
            it + 1, loss,
            loop.state.bke_history[-1],
            loop.state.boundary_history[-1],
        )

        if method == "opes":
            diag = loop.diagnostics
            logger.info(
                "  OPES: n_kernels=%d, rct=%.4f, ess=%.4f",
                diag["n_kernels"], diag["rct"], diag["ess"],
            )

        if (it + 1) % checkpoint_interval == 0:
            ckpt_path = out / f"checkpoint_{it+1:07d}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": loop.optimizer.state_dict(),
                "iteration": it + 1,
                "loss_history": loop.state.loss_history,
                "bke_history": loop.state.bke_history,
                "boundary_history": loop.state.boundary_history,
            }, ckpt_path)
            logger.info("Saved checkpoint to %s", ckpt_path)

    # Save final model
    final_path = out / "committor_final.pt"
    torch.save(model.state_dict(), final_path)
    logger.info("Training complete. Final model saved to %s", final_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train committor for CLN025 protein folding"
    )
    parser.add_argument("--system_xml", required=True,
                        help="Serialized OpenMM System XML")
    parser.add_argument("--pdb_folded", required=True,
                        help="Folded state PDB (solvated; same atoms as system.xml)")
    parser.add_argument("--pdb_unfolded", required=True,
                        help="Unfolded state PDB (solvated; same atoms as system.xml)")
    parser.add_argument(
        "--reference_pdb",
        default=None,
        help="PDB defining Simulation topology (default: same as --pdb_folded). "
        "Must match system.xml atom count.",
    )
    parser.add_argument(
        "--foundation_model",
        default="mace-off-small",
        help=(
            "Foundation model specifier. Pre-trained MACE models: "
            "'mace-off-small' (default, organic/bio), 'mace-off-medium', "
            "'mace-off-large', 'mace-mp-small' (materials), etc. "
            "Test-only: 'distance_encoder'."
        ),
    )
    parser.add_argument("--checkpoint_path", default=None,
                        help="Custom MACE checkpoint path (overrides auto-download)")
    parser.add_argument("--method", default="opes",
                        choices=["opes", "umbrella"])
    parser.add_argument("--n_iterations", type=int, default=100)
    parser.add_argument("--n_sim_steps", type=int, default=1000)
    parser.add_argument("--collect_interval", type=int, default=100)
    parser.add_argument(
        "--kernel_deposit_interval",
        type=int,
        default=50,
        help="(OPES) deposit kernel every this many steps",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--platform", default="CPU",
                        choices=["CPU", "CUDA", "OpenCL"])
    parser.add_argument("--temperature", type=float, default=340.0)
    parser.add_argument("--vk_lambda", type=float, default=1.0)
    parser.add_argument("--opes_barrier", type=float, default=5.0)
    parser.add_argument("--opes_sigma", type=float, default=0.5)
    parser.add_argument(
        "--bias_type",
        default="umbrella",
        choices=["umbrella", "kolmogorov", "none"],
        help=(
            "TorchForce bias type for OpenMM MD. 'umbrella' (default) applies a "
            "harmonic bias on z toward 0.5, requires only first-order gradients, "
            "and is CUDA-JIT-safe for any system size. 'kolmogorov' applies V_K "
            "which requires second-order gradients and may be slow for large systems. "
            "'none' disables the NN bias (unbiased MD)."
        ),
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=8,
        help=(
            "Hidden dimension for DistanceEncoder MLP. For a solvated protein with "
            "N atoms, the first Linear layer is N*(N-1)/2 → hidden_dim, so keep this "
            "small (e.g. 8) to avoid GPU OOM. Larger values increase model capacity "
            "but scale quadratically in GPU memory with atom count."
        ),
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=128,
        help="MACE encoder output / committor-head input dimension (default: 128).",
    )
    parser.add_argument(
        "--require_cuda",
        action="store_true",
        help=(
            "Require PyTorch CUDA and --platform CUDA. Exit with an error if no GPU. "
            "Use for production GPU-only runs."
        ),
    )
    parser.add_argument(
        "--strict_openmm_platform",
        action="store_true",
        help=(
            "Do not fall back to OpenCL/CPU if OpenMM cannot create a Context on "
            "--platform (e.g. keep CUDA-only)."
        ),
    )
    parser.add_argument(
        "--resume_from",
        default=None,
        help="Path to a checkpoint .pt file to resume training from.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    train(
        system_xml=args.system_xml,
        pdb_folded=args.pdb_folded,
        pdb_unfolded=args.pdb_unfolded,
        reference_pdb=args.reference_pdb,
        foundation_model=args.foundation_model,
        checkpoint_path=args.checkpoint_path,
        method=args.method,
        n_iterations=args.n_iterations,
        n_sim_steps=args.n_sim_steps,
        collect_interval=args.collect_interval,
        kernel_deposit_interval=args.kernel_deposit_interval,
        lr=args.lr,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.output_dir,
        platform=args.platform,
        temperature_K=args.temperature,
        vk_lambda=args.vk_lambda,
        opes_barrier=args.opes_barrier,
        opes_sigma=args.opes_sigma,
        bias_type=args.bias_type,
        hidden_dim=args.hidden_dim,
        require_cuda=args.require_cuda,
        strict_openmm_platform=args.strict_openmm_platform,
        feature_dim=args.feature_dim,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
