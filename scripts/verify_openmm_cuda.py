#!/usr/bin/env python3
"""Check whether the current env can run OpenMM on CUDA (and why not if PTX 222).

Run: python scripts/verify_openmm_cuda.py
"""

from __future__ import annotations

import re
import subprocess
import sys


def main() -> int:
    try:
        import openmm
    except Exception as e:  # noqa: BLE001
        print("Could not import openmm:", e, file=sys.stderr)
        return 1

    print("openmm", getattr(openmm, "__version__", "?"))
    libp = openmm.version.openmm_library_path
    print("lib:", libp() if callable(libp) else libp)

    try:
        out = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        m = re.search(
            r"Driver Version:\s*(\S+).*?CUDA Version:\s*(\S+)",
            out.stdout,
            re.DOTALL,
        )
        if m:
            print("Driver:", m.group(1), "  nvidia-smi CUDA Version:", m.group(2))
        else:
            print("nvidia-smi (partial):\n", out.stdout[:500])
    except FileNotFoundError:
        print("nvidia-smi: not in PATH (no NVIDIA driver?)")
    except Exception as e:  # noqa: BLE001
        print("nvidia-smi error:", e)

    print("\nPlatforms:")
    n = openmm.Platform.getNumPlatforms()
    for i in range(n):
        p = openmm.Platform.getPlatform(i)
        print(f"  {i} {p.getName()}")

    print("\nTiny CUDA context test (1 particle):")
    try:
        import openmm.unit as u

        s = openmm.System()
        s.addParticle(1.0 * u.dalton)
        integ = openmm.VerletIntegrator(1.0 * u.femtoseconds)
        p = openmm.Platform.getPlatformByName("CUDA")
        ctx = openmm.Context(s, integ, p)
        # Context creation compiles/loads CUDA modules (PTX) — do not need positions for that
        del ctx
        print("  OK — CUDA Context created (kernels/PTX path exercised)")
    except openmm.OpenMMException as e:
        err = str(e)
        print("  FAILED:", err)
        if "PTX" in err or "222" in err or "UNSUPPORTED" in err.upper():
            print(
                "\nThis is usually: OpenMM’s CUDA build needs a *newer* driver than the "
                "PTXs your GPU driver can JIT, or a mismatched cudatoolkit/nvrtc.\n"
                "Try:\n"
                "  1) Upgrade NVIDIA driver (production branch for your GPU).\n"
                "  2) In the same conda env, align CUDA with your driver, e.g.:\n"
                "     conda install -c conda-forge cudatoolkit=11.8 \"openmm>=8.2,<8.3\"\n"
                "  For conda-forge OpenMM 8.5 you typically need a driver whose "
                "‘CUDA Version’ in nvidia-smi is 12.9+.\n"
                "  See the repository README section on OpenMM CUDA (Linux).",
            )
        return 2
    except Exception as e:  # noqa: BLE001
        print("  error:", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
