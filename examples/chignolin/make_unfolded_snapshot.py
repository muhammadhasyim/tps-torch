"""Generate an unfolded PDB for chignolin from a minimized reference.

Loads ``system.xml`` and positions from a matching PDB, runs short
high-temperature MD to unfold the protein, and writes a new PDB.

Usage
-----
    python make_unfolded_snapshot.py \\
        --input_pdb output/minimized.pdb \\
        --system_xml output/system.xml \\
        --output output/unfolded.pdb
"""

from __future__ import annotations

import argparse
import logging
import sys

from conda_ensure import ensure_conda_lib_path_early

ensure_conda_lib_path_early()

try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit

    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False


def run() -> None:
    p = argparse.ArgumentParser(
        description="High-T MD snapshot as unfolded PDB for chignolin"
    )
    p.add_argument(
        "--input_pdb",
        required=True,
        help="PDB whose topology matches system.xml (e.g. output/minimized.pdb)",
    )
    p.add_argument(
        "--system_xml",
        required=True,
        help="Serialized system from the same setup_system.py run",
    )
    p.add_argument(
        "--output",
        default="output/unfolded.pdb",
        help="PDB to write (default: output/unfolded.pdb)",
    )
    p.add_argument(
        "--n_steps",
        type=int,
        default=30_000,
        help="MD steps at target temperature",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=450.0,
        help="Target temperature in K for unfolding",
    )
    p.add_argument(
        "--warmup_steps",
        type=int,
        default=10_000,
        help="NVT equilibration at warmup temperature before heating",
    )
    p.add_argument(
        "--warmup_temperature",
        type=float,
        default=340.0,
        help="Equilibration temperature in K",
    )
    p.add_argument(
        "--platform",
        default="CPU",
        choices=["CPU", "CUDA", "OpenCL"],
    )
    p.add_argument(
        "--timestep_fs",
        type=float,
        default=1.0,
        help="Langevin timestep in fs",
    )
    p.add_argument(
        "--keep_barostat",
        action="store_true",
        help="Keep MonteCarloBarostat if present",
    )
    args = p.parse_args()
    if not HAS_OPENMM:
        sys.exit("OpenMM is required: conda install -c conda-forge openmm")
    logging.basicConfig(level=logging.INFO)

    from committorch.openmm_platforms import create_simulation_with_platform_fallback

    pdb_in = app.PDBFile(args.input_pdb)
    with open(args.system_xml) as f:
        system = openmm.XmlSerializer.deserialize(f.read())
    n_sys = system.getNumParticles()
    n_pdb = pdb_in.topology.getNumAtoms()
    if n_sys != n_pdb:
        raise SystemExit(
            f"Atom count mismatch: {args.input_pdb!r} has {n_pdb} atoms, "
            f"system has {n_sys} particles."
        )

    if not args.keep_barostat:
        for i in range(system.getNumForces() - 1, -1, -1):
            if isinstance(system.getForce(i), openmm.MonteCarloBarostat):
                system.removeForce(i)
                logging.info("Removed MonteCarloBarostat.")

    ts = args.timestep_fs * unit.femtoseconds
    gamma = 1.0 / unit.picosecond
    t_warm = args.warmup_temperature * unit.kelvin
    t_hot = args.temperature * unit.kelvin
    log = logging.getLogger(__name__)

    int_w = openmm.LangevinMiddleIntegrator(t_warm, gamma, ts)
    sim = create_simulation_with_platform_fallback(
        pdb_in.topology, system, int_w, args.platform, log=log,
    )
    sim.context.setPositions(pdb_in.positions)
    logging.info("Minimizing energy...")
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(t_warm)
    logging.info("Warm-up: %d steps at %s K", args.warmup_steps, args.warmup_temperature)
    sim.step(args.warmup_steps)

    state0 = sim.context.getState(getPositions=True, getVelocities=True)
    int_h = openmm.LangevinMiddleIntegrator(t_hot, gamma, ts)
    sim2 = create_simulation_with_platform_fallback(
        pdb_in.topology, system, int_h, args.platform, log=log,
    )
    sim2.context.setPositions(state0.getPositions())
    sim2.context.setVelocitiesToTemperature(t_hot)
    logging.info("Production: %d steps at %s K ...", args.n_steps, args.temperature)
    sim2.step(args.n_steps)
    state = sim2.context.getState(getPositions=True)
    with open(args.output, "w") as f:
        app.PDBFile.writeFile(sim2.topology, state.getPositions(), f)
    logging.info("Wrote %s", args.output)


if __name__ == "__main__":
    run()
