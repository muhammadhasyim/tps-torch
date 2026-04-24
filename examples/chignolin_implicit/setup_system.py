"""Build an OpenMM System for CLN025 (chignolin, 10 residues).

Creates a solvated, equilibrated system from a PDB file using
CHARMM22* or Amber ff14SB force field with TIP3P water at 340 K
to match the DESRES simulation conditions.

Outputs a serialized OpenMM System XML and a minimized PDB.

Usage
-----
    python setup_system.py --pdb CLN025.pdb [--forcefield amber] [--output_dir output]

Data requirements
-----------------
- CLN025 native PDB: 5AWL from PDB or from DESRES dataset
  (download from https://www.rcsb.org/structure/5AWL)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from conda_ensure import ensure_conda_lib_path_early

ensure_conda_lib_path_early()

logger = logging.getLogger(__name__)

try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    raise ImportError(
        "OpenMM is required for this script. "
        "Install with: conda install -c conda-forge openmm"
    )

try:
    from pdbfixer import PDBFixer
    HAS_PDBFIXER = True
except ImportError:
    HAS_PDBFIXER = False
    PDBFixer = None  # type: ignore[misc, assignment]

from committorch.openmm_platforms import create_simulation_with_platform_fallback


def build_cln025_system(
    pdb_path: str,
    forcefield: str = "amber",
    solvent_model: str = "explicit",
    temperature_K: float = 340.0,
    padding_nm: float = 1.0,
    nonbonded_cutoff_nm: float = 1.0,
    ionic_strength_M: float = 0.0,
    hydrogen_mass_amu: float = 1.5,
    timestep_fs: float = 2.0,
    platform: str = "CPU",
    use_pdbfixer: bool = True,
    ph: float = 7.0,
    strict_openmm_platform: bool = False,
) -> tuple[app.Simulation, openmm.System]:
    """Build and minimize a CLN025 system.

    Parameters
    ----------
    pdb_path : str
        Path to the CLN025 PDB file.
    forcefield : str
        "amber" for Amber ff14SB or "charmm" for CHARMM22*.
    solvent_model : str
        Solvent treatment:

        - ``"explicit"`` (default): TIP3P water box with PME electrostatics.
          Yields ~4800 atoms for CLN025.
        - ``"implicit"``: GBn2 (Generalized Born) implicit solvent.
          Protein-only, ~166 atoms.  Much more GPU-friendly for MLIP
          backbone models.
        - ``"vacuum"``: No solvent at all.  Suitable for gas-phase tests.
    temperature_K : float
        Simulation temperature in Kelvin (340 K matches DESRES).
    padding_nm : float
        Solvent padding around the protein (explicit only).
    nonbonded_cutoff_nm : float
        Cutoff for nonbonded interactions (explicit only).
    ionic_strength_M : float
        Ionic strength in mol/L (explicit only).
    hydrogen_mass_amu : float
        Hydrogen mass for HMR (enables longer timesteps).
    timestep_fs : float
        Integration timestep in femtoseconds.
    platform : str
        Requested platform: "CUDA", "OpenCL", or "CPU".
    use_pdbfixer : bool
        If True (default), use PDBFixer to find missing atoms/residues.
    ph : float
        pH for protonation in PDBFixer / Modeller.
    strict_openmm_platform : bool
        If True, do not fall back to OpenCL/CPU when the requested
        platform fails (e.g. use GPU-only runs with ``--platform CUDA``).

    Returns
    -------
    simulation : openmm.app.Simulation
    system : openmm.System
    """
    if solvent_model == "implicit":
        if forcefield == "amber":
            ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
        elif forcefield == "charmm":
            ff = app.ForceField("charmm36.xml", "implicit/gbn2.xml")
        else:
            raise ValueError(f"Unknown forcefield: {forcefield}")
    elif solvent_model in ("explicit", "vacuum"):
        if forcefield == "amber":
            water_xml = "amber14/tip3pfb.xml" if solvent_model == "explicit" else ""
            ff_files = ["amber14-all.xml"]
            if water_xml:
                ff_files.append(water_xml)
            ff = app.ForceField(*ff_files)
        elif forcefield == "charmm":
            ff_files = ["charmm36.xml"]
            if solvent_model == "explicit":
                ff_files.append("charmm36/water.xml")
            ff = app.ForceField(*ff_files)
        else:
            raise ValueError(f"Unknown forcefield: {forcefield}")
    else:
        raise ValueError(
            f"Unknown solvent_model '{solvent_model}'. "
            f"Choose 'explicit', 'implicit', or 'vacuum'."
        )

    if use_pdbfixer:
        if not HAS_PDBFIXER:
            raise ImportError(
                "PDBFixer is required for use_pdbfixer=True. "
                "Install with: pip install pdbfixer  "
                "or conda install -c conda-forge pdbfixer  "
                "or pass use_pdbfixer=False to use only Modeller.addHydrogens."
            )
        logger.info("PDBFixer: find missing residues/atoms, add hydrogens (pH=%s)...", ph)
        fixer = PDBFixer(filename=str(pdb_path))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(ph, ff)
        modeller = app.Modeller(fixer.topology, fixer.positions)
    else:
        pdb = app.PDBFile(pdb_path)
        modeller = app.Modeller(pdb.topology, pdb.positions)
        logger.info("Modeller.addHydrogens only (pH=%s) — use PDBFixer for missing atoms", ph)
        modeller.addHydrogens(ff, pH=ph)

    if solvent_model == "explicit":
        modeller.addSolvent(
            ff,
            model="tip3p",
            padding=padding_nm * unit.nanometers,
            ionicStrength=ionic_strength_M * unit.molar,
        )
        nonbonded_method = app.PME
        logger.info("Added explicit TIP3P solvent (padding=%.1f nm)", padding_nm)
    elif solvent_model == "implicit":
        nonbonded_method = app.NoCutoff
        logger.info("Using GBn2 implicit solvent (no explicit water)")
    else:
        nonbonded_method = app.NoCutoff
        logger.info("Vacuum (no solvent)")

    create_system_kwargs: dict[str, Any] = {
        "nonbondedMethod": nonbonded_method,
        "constraints": app.HBonds,
        "hydrogenMass": hydrogen_mass_amu * unit.amu,
    }
    if solvent_model == "explicit":
        create_system_kwargs["nonbondedCutoff"] = nonbonded_cutoff_nm * unit.nanometers

    system = ff.createSystem(modeller.topology, **create_system_kwargs)

    temperature = temperature_K * unit.kelvin
    integrator = openmm.LangevinMiddleIntegrator(
        temperature,
        1.0 / unit.picosecond,
        timestep_fs * unit.femtoseconds,
    )

    if solvent_model == "explicit":
        system.addForce(openmm.MonteCarloBarostat(
            1.0 * unit.atmosphere, temperature
        ))

    simulation = create_simulation_with_platform_fallback(
        modeller.topology,
        system,
        integrator,
        platform,
        log=logger,
        strict=strict_openmm_platform,
    )
    simulation.context.setPositions(modeller.positions)

    logger.info("Minimizing energy...")
    simulation.minimizeEnergy()

    n_atoms = simulation.topology.getNumAtoms()
    logger.info("System has %d atoms (solvent_model=%s)", n_atoms, solvent_model)

    logger.info("Equilibrating for 10 ps...")
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.step(5000)

    return simulation, system


def save_system(
    simulation: app.Simulation,
    system: openmm.System,
    output_dir: str = "output",
) -> None:
    """Save the serialized system and minimized positions.

    Parameters
    ----------
    simulation : openmm.app.Simulation
    system : openmm.System
    output_dir : str
        Output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    system_xml = openmm.XmlSerializer.serialize(system)
    (out / "system.xml").write_text(system_xml)

    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions()
    with open(out / "minimized.pdb", "w") as f:
        app.PDBFile.writeFile(simulation.topology, positions, f)

    logger.info("Saved system to %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CLN025 OpenMM system"
    )
    parser.add_argument(
        "--pdb", required=True,
        help="Path to CLN025 PDB file (e.g. 5AWL.pdb)"
    )
    parser.add_argument(
        "--forcefield", default="amber",
        choices=["amber", "charmm"],
        help="Force field (default: amber)"
    )
    parser.add_argument(
        "--solvent_model",
        default="explicit",
        choices=["explicit", "implicit", "vacuum"],
        help=(
            "Solvent treatment: 'explicit' (TIP3P, ~4800 atoms), "
            "'implicit' (GBn2, protein-only ~166 atoms), or "
            "'vacuum' (no solvent). Default: explicit."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=340.0,
        help="Temperature in Kelvin (default: 340.0)"
    )
    parser.add_argument(
        "--platform", default="CPU",
        choices=["CPU", "CUDA", "OpenCL"],
        help="OpenMM platform"
    )
    parser.add_argument(
        "--output_dir", default="output",
        help="Output directory"
    )
    parser.add_argument(
        "--no-pdbfixer",
        action="store_true",
        help="Skip PDBFixer; use Modeller.addHydrogens only (no missing heavy-atom fixup)",
    )
    parser.add_argument(
        "--ph", type=float, default=7.0,
        help="pH for protonation (default: 7.0)",
    )
    parser.add_argument(
        "--strict_openmm_platform",
        action="store_true",
        help="Fail if OpenMM cannot use the requested --platform (no OpenCL/CPU fallback).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    simulation, system = build_cln025_system(
        pdb_path=args.pdb,
        forcefield=args.forcefield,
        solvent_model=args.solvent_model,
        temperature_K=args.temperature,
        platform=args.platform,
        use_pdbfixer=not args.no_pdbfixer,
        ph=args.ph,
        strict_openmm_platform=args.strict_openmm_platform,
    )

    save_system(simulation, system, args.output_dir)
    logger.info("CLN025 system setup complete.")


if __name__ == "__main__":
    main()
