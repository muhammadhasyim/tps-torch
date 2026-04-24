"""Build an OpenMM System for chignolin (1UAO, 10 residues).

Creates a protein system in vacuum, implicit, or explicit solvent from a
PDB file using Amber ff14SB, minimized and equilibrated.  The implicit
solvent path (GBn2) yields ~138 atoms -- small enough for MACE-OFF-small
on a 12 GB GPU.

Outputs a serialized OpenMM System XML and a minimized PDB.

Usage
-----
    # Download 1UAO.pdb first (from RCSB or use --fetch_pdb)
    python setup_system.py --pdb 1UAO.pdb --solvent_model implicit

    # Auto-fetch from RCSB:
    python setup_system.py --fetch_pdb 1UAO --solvent_model implicit

Data requirements
-----------------
- Chignolin PDB: 1UAO from PDB
  (download from https://www.rcsb.org/structure/1UAO)
"""

from __future__ import annotations

import argparse
import logging
import urllib.request
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


def fetch_pdb(pdb_id: str, output_path: str) -> str:
    """Download a PDB file from RCSB.

    Parameters
    ----------
    pdb_id : str
        4-character PDB ID (e.g. ``"1UAO"``).
    output_path : str
        File path to write (e.g. ``"1UAO.pdb"``).

    Returns
    -------
    str
        Path to the downloaded file.
    """
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    logger.info("Downloading %s from RCSB...", url)
    urllib.request.urlretrieve(url, output_path)
    logger.info("Saved to %s", output_path)
    return output_path


def build_chignolin_system(
    pdb_path: str,
    forcefield: str = "amber",
    solvent_model: str = "implicit",
    temperature_K: float = 340.0,
    padding_nm: float = 1.0,
    nonbonded_cutoff_nm: float = 1.0,
    ionic_strength_M: float = 0.0,
    hydrogen_mass_amu: float = 1.5,
    timestep_fs: float = 2.0,
    platform: str = "CPU",
    use_pdbfixer: bool = True,
    ph: float = 7.0,
) -> tuple[app.Simulation, openmm.System]:
    """Build and minimize a chignolin system.

    Parameters
    ----------
    pdb_path : str
        Path to the chignolin PDB file.
    forcefield : str
        ``"amber"`` for Amber ff14SB.
    solvent_model : str
        ``"implicit"`` (GBn2, default, ~138 atoms),
        ``"explicit"`` (TIP3P water box), or ``"vacuum"``.
    temperature_K : float
        Simulation temperature in Kelvin.
    padding_nm : float
        Solvent padding around the protein (explicit only).
    nonbonded_cutoff_nm : float
        Cutoff for nonbonded interactions (explicit only).
    ionic_strength_M : float
        Ionic strength in mol/L (explicit only).
    hydrogen_mass_amu : float
        Hydrogen mass for HMR.
    timestep_fs : float
        Integration timestep in femtoseconds.
    platform : str
        OpenMM platform.
    use_pdbfixer : bool
        If True, use PDBFixer to fix missing atoms.
    ph : float
        pH for protonation.

    Returns
    -------
    simulation : openmm.app.Simulation
    system : openmm.System
    """
    if solvent_model == "implicit":
        if forcefield == "amber":
            ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
        else:
            raise ValueError(f"Unsupported forcefield for implicit solvent: {forcefield}")
    elif solvent_model in ("explicit", "vacuum"):
        if forcefield == "amber":
            ff_files = ["amber14-all.xml"]
            if solvent_model == "explicit":
                ff_files.append("amber14/tip3pfb.xml")
            ff = app.ForceField(*ff_files)
        else:
            raise ValueError(f"Unsupported forcefield: {forcefield}")
    else:
        raise ValueError(
            f"Unknown solvent_model '{solvent_model}'. "
            f"Choose 'explicit', 'implicit', or 'vacuum'."
        )

    if use_pdbfixer:
        if not HAS_PDBFIXER:
            raise ImportError(
                "PDBFixer is required. Install: conda install -c conda-forge pdbfixer"
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
        system.addForce(
            openmm.MonteCarloBarostat(1.0 * unit.atmosphere, temperature)
        )

    simulation = create_simulation_with_platform_fallback(
        modeller.topology,
        system,
        integrator,
        platform,
        log=logger,
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
    """Save the serialized system and minimized positions."""
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
        description="Build chignolin (1UAO) OpenMM system"
    )
    pdb_group = parser.add_mutually_exclusive_group(required=True)
    pdb_group.add_argument(
        "--pdb",
        help="Path to chignolin PDB file (e.g. 1UAO.pdb)",
    )
    pdb_group.add_argument(
        "--fetch_pdb",
        metavar="PDB_ID",
        help="4-character PDB ID to auto-download from RCSB (e.g. 1UAO)",
    )
    parser.add_argument(
        "--forcefield",
        default="amber",
        choices=["amber"],
        help="Force field (default: amber)",
    )
    parser.add_argument(
        "--solvent_model",
        default="implicit",
        choices=["explicit", "implicit", "vacuum"],
        help=(
            "Solvent treatment: 'implicit' (GBn2, ~138 atoms, default), "
            "'explicit' (TIP3P water box), or 'vacuum' (gas phase)."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=340.0,
        help="Temperature in Kelvin (default: 340.0)",
    )
    parser.add_argument(
        "--platform",
        default="CPU",
        choices=["CPU", "CUDA", "OpenCL"],
        help="OpenMM platform",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help="Output directory",
    )
    parser.add_argument(
        "--no-pdbfixer",
        action="store_true",
        help="Skip PDBFixer; use Modeller.addHydrogens only",
    )
    parser.add_argument(
        "--ph",
        type=float,
        default=7.0,
        help="pH for protonation (default: 7.0)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.fetch_pdb:
        pdb_path = fetch_pdb(args.fetch_pdb, f"{args.fetch_pdb.upper()}.pdb")
    else:
        pdb_path = args.pdb

    simulation, system = build_chignolin_system(
        pdb_path=pdb_path,
        forcefield=args.forcefield,
        solvent_model=args.solvent_model,
        temperature_K=args.temperature,
        platform=args.platform,
        use_pdbfixer=not args.no_pdbfixer,
        ph=args.ph,
    )

    save_system(simulation, system, args.output_dir)
    logger.info("Chignolin system setup complete.")


if __name__ == "__main__":
    main()
