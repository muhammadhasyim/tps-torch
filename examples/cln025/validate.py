"""Validate a trained committor against trajectory data (e.g. DESRES).

Loads a trained committor (same ``foundation_model`` / topology as training),
evaluates ``q(x)`` along trajectories, and compares RMSD-based state labels.

Usage
-----
    python validate.py --model_path output/committor_final.pt \\
        --reference_pdb output/minimized.pdb \\
        --desres_dir /path/to/trajectories \\
        --foundation_model mace-off-small

``reference_pdb`` must match the atom count and order used during training
(typically ``output/minimized.pdb`` from ``setup_system.py``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _ensure_imports() -> None:
    try:
        from conda_ensure import ensure_conda_lib_path_early

        ensure_conda_lib_path_early()
    except ImportError:
        repo = Path(__file__).resolve().parent.parent.parent
        src = repo / "src"
        if src.is_dir():
            sys.path.insert(0, str(src))


_ensure_imports()


def _get_atomic_numbers_from_topology(topology) -> torch.Tensor:
    nums = []
    for atom in topology.atoms():
        el = atom.element
        if el is not None:
            nums.append(el.atomic_number)
        else:
            nums.append(0)
    return torch.tensor(nums, dtype=torch.long)


def load_committor_model(
    model_path: str,
    foundation_model: str = "mace-off-small",
    reference_pdb: str | None = None,
    n_atoms: int | None = None,
    checkpoint_path: str | None = None,
    feature_dim: int = 128,
    hidden_dim: int = 8,
    device: str = "cpu",
) -> torch.nn.Module:
    """Rebuild the committor and load weights from ``model_path``."""
    from committorch.models.pretrained import load_pretrained
    from openmm.app import PDBFile

    if foundation_model == "distance_encoder":
        if n_atoms is None:
            raise ValueError("n_atoms is required when foundation_model is distance_encoder")
        model = load_pretrained(
            "distance_encoder",
            n_atoms=n_atoms,
            hidden_dim=hidden_dim,
            freeze_encoder=False,
        )
    else:
        if not reference_pdb:
            raise ValueError(
                "reference_pdb is required for MACE foundation models (same topology as training)."
            )
        ref = PDBFile(reference_pdb)
        atomic_numbers = _get_atomic_numbers_from_topology(ref.topology)
        n_ref = ref.topology.getNumAtoms()
        if n_atoms is not None and n_ref != n_atoms:
            raise ValueError(
                f"reference_pdb has {n_ref} atoms but --n_atoms={n_atoms}"
            )
        if checkpoint_path is not None:
            model = load_pretrained(
                "mace",
                checkpoint_path=checkpoint_path,
                atomic_numbers=atomic_numbers,
                feature_dim=feature_dim,
                freeze_encoder=False,
                device=device,
            )
        else:
            model = load_pretrained(
                foundation_model,
                atomic_numbers=atomic_numbers,
                feature_dim=feature_dim,
                freeze_encoder=False,
                device=device,
            )

    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_committor_along_trajectory(
    model: torch.nn.Module,
    trajectory_path: str,
    topology_path: str | None = None,
    stride: int = 100,
    ca_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate q(x) for each frame in a trajectory.

    Returns
    -------
    times, committors
    """
    try:
        import mdtraj
    except ImportError as e:
        raise ImportError("mdtraj is required: pip install mdtraj") from e

    traj = mdtraj.load(trajectory_path, top=topology_path, stride=stride)

    if ca_only:
        ca_indices = traj.topology.select("name CA")
        traj = traj.atom_slice(ca_indices)

    n_frames = len(traj)
    committors = np.zeros(n_frames)
    device = next(model.parameters()).device

    with torch.no_grad():
        for i in range(n_frames):
            pos = torch.tensor(traj.xyz[i].reshape(1, -1), dtype=torch.float32, device=device)
            q = model(pos).item()
            committors[i] = q

    times = np.arange(n_frames) * stride
    if hasattr(traj, "time") and traj.time is not None:
        times = traj.time[::stride] if len(traj.time) > n_frames else traj.time

    return times, committors


def compute_brute_force_rate(
    rmsd_folded: np.ndarray,
    rmsd_cutoff: float = 0.3,
    dt_ns: float = 0.2,
) -> dict[str, float]:
    """Compute folding rate from RMSD-based state assignments."""
    folded = rmsd_folded < rmsd_cutoff
    transitions = np.where(np.diff(folded.astype(int)) != 0)[0]

    n_transitions = len(transitions)
    total_time = len(rmsd_folded) * dt_ns

    rate = n_transitions / total_time if total_time > 0 else 0.0
    mfpt = total_time / max(n_transitions, 1)

    return {
        "n_transitions": n_transitions,
        "total_time_ns": total_time,
        "rate_per_ns": rate,
        "mfpt_ns": mfpt,
    }


def validate_committor(
    model_path: str,
    desres_dir: str,
    reference_pdb: str,
    foundation_model: str = "mace-off-small",
    n_atoms: int | None = None,
    checkpoint_path: str | None = None,
    feature_dim: int = 128,
    hidden_dim: int = 8,
    stride: int = 100,
    rmsd_cutoff: float = 0.3,
    output_dir: str = "validation_output",
    require_cuda: bool = False,
) -> dict:
    """Full validation pipeline.

    The trajectory topology must have the same number of atoms as ``reference_pdb``
    (training topology). External datasets (e.g. DESRES) often differ; use
    trajectories built from the same OpenMM system as training when possible.
    """
    import mdtraj
    from openmm.app import PDBFile

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "--require_cuda set but torch.cuda.is_available() is False."
        )
    dev = "cuda" if (require_cuda or torch.cuda.is_available()) else "cpu"
    if require_cuda and dev != "cuda":
        raise RuntimeError("CUDA required but not available.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = load_committor_model(
        model_path,
        foundation_model=foundation_model,
        reference_pdb=reference_pdb,
        n_atoms=n_atoms,
        checkpoint_path=checkpoint_path,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        device=dev,
    )
    if require_cuda or dev == "cuda":
        model = model.cuda()
    logger.info("Loaded model from %s (device=%s)", model_path, next(model.parameters()).device)

    n_expected = PDBFile(reference_pdb).topology.getNumAtoms()

    desres_path = Path(desres_dir)
    traj_files = sorted(desres_path.glob("*.dcd")) + sorted(desres_path.glob("*.xtc"))
    if not traj_files:
        raise FileNotFoundError(f"No trajectory files found in {desres_dir}")

    top_files = sorted(desres_path.glob("*.pdb"))
    if not top_files:
        raise FileNotFoundError(f"No PDB topology found in {desres_dir}")

    topology_path = str(top_files[0])
    logger.info("Using trajectory topology: %s", topology_path)

    traj_probe = mdtraj.load(str(traj_files[0]), top=topology_path, stride=stride)
    if traj_probe.n_atoms != n_expected:
        raise ValueError(
            f"Atom count mismatch: reference_pdb has {n_expected} atoms but "
            f"trajectory (with top {topology_path}) has {traj_probe.n_atoms} atoms. "
            "Use trajectories with the same system as training, or retrain to match."
        )

    native = mdtraj.load(topology_path)

    all_committors: list[np.ndarray] = []
    all_rmsds: list[np.ndarray] = []

    for traj_file in traj_files:
        logger.info("Processing %s", traj_file.name)
        traj = mdtraj.load(str(traj_file), top=topology_path, stride=stride)

        ca_idx = traj.topology.select("name CA")
        rmsds = mdtraj.rmsd(traj, native, atom_indices=ca_idx)
        all_rmsds.append(rmsds)

        _, committors = evaluate_committor_along_trajectory(
            model, str(traj_file), topology_path, stride=stride
        )
        all_committors.append(committors)

    all_committors = np.concatenate(all_committors)
    all_rmsds = np.concatenate(all_rmsds)

    folded_mask = all_rmsds < rmsd_cutoff
    committor_folded = all_committors < 0.5
    accuracy = float(np.mean(folded_mask == committor_folded))
    logger.info("State assignment accuracy: %.2f%%", 100 * accuracy)

    bf_rate = compute_brute_force_rate(all_rmsds, rmsd_cutoff)
    logger.info("Brute-force rate: %.4e per ns", bf_rate["rate_per_ns"])
    logger.info("MFPT: %.1f ns", bf_rate["mfpt_ns"])

    results = {
        "accuracy": accuracy,
        "brute_force_rate": bf_rate,
        "n_frames": len(all_committors),
        "mean_committor_folded": float(all_committors[folded_mask].mean())
        if folded_mask.any()
        else None,
        "mean_committor_unfolded": float(all_committors[~folded_mask].mean())
        if (~folded_mask).any()
        else None,
    }

    with open(out / "validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s", out / "validation_results.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate committor against trajectories (e.g. DESRES)"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--reference_pdb",
        required=True,
        help="PDB with same atom count/order as training (e.g. output/minimized.pdb).",
    )
    parser.add_argument("--desres_dir", required=True, help="Directory with *.dcd/*.xtc and *.pdb")
    parser.add_argument(
        "--foundation_model",
        default="mace-off-small",
        help="Same as training (e.g. mace-off-small or distance_encoder).",
    )
    parser.add_argument(
        "--n_atoms",
        type=int,
        default=None,
        help="Only for distance_encoder: number of atoms.",
    )
    parser.add_argument("--checkpoint_path", default=None, help="Custom MACE .model path (optional)")
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=8)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--rmsd_cutoff", type=float, default=0.3)
    parser.add_argument("--output_dir", default="validation_output")
    parser.add_argument(
        "--require_cuda",
        action="store_true",
        help="Run the committor on CUDA only.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    validate_committor(
        model_path=args.model_path,
        desres_dir=args.desres_dir,
        reference_pdb=args.reference_pdb,
        foundation_model=args.foundation_model,
        n_atoms=args.n_atoms,
        checkpoint_path=args.checkpoint_path,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        stride=args.stride,
        rmsd_cutoff=args.rmsd_cutoff,
        output_dir=args.output_dir,
        require_cuda=args.require_cuda,
    )


if __name__ == "__main__":
    main()
