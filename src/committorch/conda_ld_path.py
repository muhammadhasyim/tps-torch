"""Conda: ensure conda's C++ runtimes are found before the system (Linux).

The dynamic linker on Linux can resolve :file:`/lib/.../libstdc++.so.6` before
``$CONDA_PREFIX/lib`` when OpenMM and other Conda packages need a newer
``CXXABI_*`` from ``libstdcxx-ng`` (see e.g. ``CXXABI_1.3.15``). Prepend
``$CONDA_PREFIX/lib`` to :envvar:`LD_LIBRARY_PATH` in the *current process* before
importing any extension that loads those shared objects.

This must be called as early as possible (before :mod:`torch`, :mod:`openmm`).

``CONDA_PREFIX`` is set by ``conda activate``; in non-conda runs this is a no-op.
"""

from __future__ import annotations

import os
import sys


def ensure_conda_lib_path() -> None:
    if not sys.platform.startswith("linux"):
        return
    prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if not prefix:
        return
    lib = os.path.join(prefix, "lib")
    if not os.path.isdir(lib):
        return
    key = "LD_LIBRARY_PATH"
    current = os.environ.get(key, "")
    parts = [p for p in current.split(os.pathsep) if p]
    if lib in parts and parts[0] == lib:
        return
    if lib in parts:
        parts = [lib] + [p for p in parts if p != lib]
    else:
        parts = [lib] + parts
    os.environ[key] = os.pathsep.join(parts)
