"""Build :class:`openmm.app.Simulation` with CUDA/OpenCL/CPU fallback on context failure.

The CUDA *platform* can load while *Context* creation still fails, e.g.
``CUDA_ERROR_UNSUPPORTED_PTX_VERSION`` when the OpenMM CUDA plugin’s embedded PTX
does not match the driver / NVRTC stack. The fallback chain matches what users
often do manually: try OpenCL, then CPU.
"""

from __future__ import annotations

import logging
from typing import Any

import openmm
import openmm.app as app

logger = logging.getLogger(__name__)


def create_simulation_with_platform_fallback(
    topology: Any,
    system: openmm.System,
    integrator: openmm.Integrator,
    requested_platform: str,
    log: logging.Logger | None = None,
    strict: bool = False,
) -> app.Simulation:
    """Create ``Simulation``; on failure of the first backend, try the next in chain.

    Parameters
    ----------
    strict : bool
        If True, only ``requested_platform`` is tried (no OpenCL/CPU fallback).
        Use with GPU-only workflows; raises if Context creation fails.
    """
    log = log or logger
    req = (requested_platform or "CPU").strip().upper()
    if strict:
        chain = (req,) if req in ("CUDA", "OPENCL", "CPU") else ("CPU",)
    else:
        chain = {
            "CUDA": ("CUDA", "OpenCL", "CPU"),
            "OPENCL": ("OpenCL", "CPU"),
            "CPU": ("CPU",),
        }.get(req, ("CPU",))
    first = chain[0]
    first_error: BaseException | None = None
    last_exc: BaseException | None = None

    for name in chain:
        try:
            platform = openmm.Platform.getPlatformByName(name)
        except Exception as e:  # noqa: BLE001 — try next name in chain
            last_exc = e
            if name == first:
                first_error = e
            log.debug("OpenMM getPlatformByName(%s) failed: %s", name, e)
            continue
        try:
            sim = app.Simulation(topology, system, integrator, platform)
        except openmm.OpenMMException as e:
            last_exc = e
            if name == first:
                first_error = e
            log.warning("OpenMM Context on %s failed: %s", name, e)
            continue
        if name != first:
            log.warning(
                "OpenMM first tried platform %s (%s). Using %s instead. "
                "For PTX / CUDA_ERROR_UNSUPPORTED_PTX_VERSION, use one stack from "
                "conda-forge (e.g. openmm, cuda-nvrtc) and a current NVIDIA driver, "
                "or pass --platform CPU for short jobs.",
                first,
                first_error,
                name,
            )
        return sim

    msg = f"Could not create OpenMM Simulation (tried: {chain}). Last error: {last_exc!r}"
    raise RuntimeError(msg) from last_exc
