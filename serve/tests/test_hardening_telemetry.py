"""NVML discovery uses documented Windows locations and fails safely when absent."""
import os
from unittest import mock

from serve.util import telemetry


def test_windows_nvml_loads_only_documented_absolute_paths():
    lib = mock.Mock()
    lib.nvmlInit_v2.return_value = 0
    lib.nvmlDeviceGetHandleByIndex_v2.return_value = 0
    with mock.patch.object(telemetry.sys, "platform", "win32"), \
            mock.patch.dict(os.environ, {"SystemRoot": r"C:\Windows", "ProgramW6432": r"C:\Program Files"}, clear=True), \
            mock.patch.object(telemetry.ctypes, "CDLL", side_effect=[OSError("missing"), lib]) as load:
        assert telemetry._Nvml().ok()
    assert [c.args[0] for c in load.call_args_list] == [
        r"C:\Windows\System32\nvml.dll", r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll"]


def test_missing_windows_nvml_is_safe():
    with mock.patch.object(telemetry.sys, "platform", "win32"), \
            mock.patch.dict(os.environ, {"SystemRoot": r"C:\Windows"}, clear=True), \
            mock.patch.object(telemetry.ctypes, "CDLL", side_effect=OSError("missing")):
        assert not telemetry._Nvml().ok()


def test_linux_nvml_discovery_is_unchanged():
    with mock.patch.object(telemetry.sys, "platform", "linux"), \
            mock.patch.object(telemetry.ctypes, "CDLL", side_effect=OSError("missing")) as load:
        assert not telemetry._Nvml().ok()
    assert [c.args[0] for c in load.call_args_list] == ["libnvidia-ml.so.1", "libnvidia-ml.so"]
