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


def test_cpu_name_closes_proc_file_on_early_return(tmp_path):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("model name : Test CPU\nunused : row\n")
    stream = cpuinfo.open()
    with mock.patch.object(telemetry.os.path, "exists", return_value=True), \
            mock.patch.object(telemetry, "open", return_value=stream):
        assert telemetry._cpu_name() == "Test CPU"
    assert stream.closed


def test_cpu_times_closes_proc_file_on_success_and_parse_failure(tmp_path):
    stat = tmp_path / "stat"
    reader = telemetry._CpuRamFallback.__new__(telemetry._CpuRamFallback)
    for content, expected in [("cpu 1 2 3 4 5 6 7 8\n", (9, 36)), ("cpu invalid\n", None)]:
        stat.write_text(content)
        stream = stat.open()
        with mock.patch.object(telemetry, "open", return_value=stream):
            assert reader._times() == expected
        assert stream.closed


def test_ram_closes_proc_file_on_success_and_parse_failure(tmp_path):
    meminfo = tmp_path / "meminfo"
    for content, expected in [("MemTotal: 100 kB\nMemAvailable: 25 kB\n", (75 * 1024, 100 * 1024)),
                              ("invalid row\n", (None, None))]:
        meminfo.write_text(content)
        stream = meminfo.open()
        with mock.patch.object(telemetry, "open", return_value=stream):
            assert telemetry._CpuRamFallback.ram() == expected
        assert stream.closed
