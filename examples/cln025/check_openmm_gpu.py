#!/usr/bin/env python3
"""Print GPU driver info and test OpenMM CUDA / OpenCL / CPU Context creation.

Run in the same conda env you use for setup_system.py:
  python check_openmm_gpu.py

If CUDA fails with CUDA_ERROR_UNSUPPORTED_PTX_VERSION, typical fixes:
- Install OpenMM from conda-forge (not pip) so nvrtc/cuda libs match: see README.
- Avoid mixing a pip OpenMM with a different CUDA toolkit on LD_LIBRARY_PATH.
- Upgrade the NVIDIA driver to the latest for your GPU.
"""
from __future__ import annotations

import subprocess
import sys

from conda_ensure import ensure_conda_lib_path_early

ensure_conda_lib_path_early()


def _nvidia_smi() -> None:
    try:
        out = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        print(out.stdout)
        if out.returncode != 0:
            print("stderr:", out.stderr, file=sys.stderr)
    except FileNotFoundError:
        print("nvidia-smi not found (no NVIDIA driver in PATH?)\n")
    except Exception as e:
        print(f"nvidia-smi error: {e}\n")


def _openmm_smoke() -> None:
    import openmm as mm
    import openmm.unit as u

    print("OpenMM version:", mm.__version__)
    print("OpenMM file:", mm.__file__)
    print("Platforms:", [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())])
    print()

    # Tiny system: one particle, no PBC — exercises CUDA kernel load
    system = mm.System()
    system.addParticle(1.0 * u.dalton)

    for name in ("CUDA", "OpenCL", "CPU"):
        try:
            p = mm.Platform.getPlatformByName(name)
        except Exception as e:
            print(f"{name:8s}  (not available: {e})")
            continue
        integ = mm.VerletIntegrator(1.0 * u.femtoseconds)
        try:
            ctx = mm.Context(system, integ, p)
            del ctx
            print(f"{name:8s}  OK — Context created")
        except Exception as e:
            print(f"{name:8s}  FAIL — {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("=== nvidia-smi ===\n")
    _nvidia_smi()
    print("=== OpenMM context smoke test ===\n")
    _openmm_smoke()
