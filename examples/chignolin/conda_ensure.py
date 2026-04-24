"""Load ``committorch.conda_ld_path`` and prepend ``$CONDA_PREFIX/lib`` (Linux + conda).

``setup_system.py`` and other chignolin scripts are often run as plain files. If
``committorch`` is not on ``PYTHONPATH`` yet, the repo ``src/`` is added so the
shared helper in the package is used.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_conda_lib_path_early() -> None:
    try:
        from committorch.conda_ld_path import ensure_conda_lib_path
    except ImportError:
        repo = Path(__file__).resolve().parent.parent.parent
        src = repo / "src"
        if src.is_dir():
            sys.path.insert(0, str(src))
        from committorch.conda_ld_path import ensure_conda_lib_path
    ensure_conda_lib_path()
