from __future__ import annotations

from unittest import mock

import openmm
import openmm.app as app

from committorch.openmm_platforms import create_simulation_with_platform_fallback


class _FakePlatform:
    def __init__(self, name: str) -> None:
        self._name = name

    def getName(self) -> str:
        return self._name


def test_cuda_context_falls_back_to_opencl() -> None:
    fake_top = object()
    fake_sys = mock.Mock()
    fake_int = mock.Mock()
    calls: list[str] = []

    def fake_get(name: str) -> _FakePlatform:
        calls.append(name)
        if name == "CUDA":
            return _FakePlatform("CUDA")
        if name == "OpenCL":
            return _FakePlatform("OpenCL")
        raise RuntimeError(name)

    sim_ok = mock.Mock(spec=app.Simulation)

    def fake_Simulation(topology, system, integrator, platform) -> app.Simulation:  # noqa: ANN001
        if platform.getName() == "CUDA":
            raise openmm.OpenMMException("CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)")
        if platform.getName() == "OpenCL":
            return sim_ok
        raise AssertionError(platform.getName())

    with mock.patch.object(openmm.Platform, "getPlatformByName", staticmethod(fake_get)):
        with mock.patch.object(app, "Simulation", staticmethod(fake_Simulation)):
            out = create_simulation_with_platform_fallback(
                fake_top, fake_sys, fake_int, "CUDA", log=mock.Mock()
            )
    assert out is sim_ok
    assert calls == ["CUDA", "OpenCL"]


def test_respects_cpu_only() -> None:
    fake_top = object()
    fake_sys = mock.Mock()
    fake_int = mock.Mock()
    sim_ok = mock.Mock(spec=app.Simulation)
    with mock.patch.object(
        openmm.Platform, "getPlatformByName", return_value=_FakePlatform("CPU")
    ):
        with mock.patch.object(app, "Simulation", return_value=sim_ok) as fsim:
            out = create_simulation_with_platform_fallback(
                fake_top, fake_sys, fake_int, "CPU"
            )
    assert out is sim_ok
    fsim.assert_called_once()
