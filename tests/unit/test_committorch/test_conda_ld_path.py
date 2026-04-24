from __future__ import annotations

import os
import sys
from unittest import mock

import pytest

from committorch.conda_ld_path import ensure_conda_lib_path


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="LD_LIBRARY_PATH logic is Linux-only"
)
def test_ensure_conda_lib_path_prepends_prefix() -> None:
    with mock.patch("os.path.isdir", return_value=True):
        with mock.patch.dict(
            os.environ,
            {"CONDA_PREFIX": "/opt/conda", "LD_LIBRARY_PATH": "/old"},
            clear=True,
        ):
            ensure_conda_lib_path()
            assert os.environ["LD_LIBRARY_PATH"] == f"/opt/conda/lib{os.pathsep}/old"


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="LD_LIBRARY_PATH logic is Linux-only"
)
def test_ensure_idempotent() -> None:
    with mock.patch.dict(os.environ, {"CONDA_PREFIX": "/opt/conda"}):
        with mock.patch("os.path.isdir", return_value=True):
            with mock.patch.dict(os.environ, {}, clear=True):
                ensure_conda_lib_path()
                first = os.environ.get("LD_LIBRARY_PATH", "")
                ensure_conda_lib_path()
                assert os.environ.get("LD_LIBRARY_PATH", "") == first


def test_ensure_noop_without_conda() -> None:
    with mock.patch.dict(os.environ, {}, clear=True):
        ensure_conda_lib_path()  # should not raise
