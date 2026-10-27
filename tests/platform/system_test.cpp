// tests/platform/system_test.cpp - the host-system questions in basalt/platform/system.hpp, the DynLib loader and the
// memory-accounting paths. CPU only. Each answer is checked for being usable; none depends on a particular machine.
#include "basalt/platform/dynlib.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/platform/process.hpp"
#include "basalt/platform/system.hpp"

#include <cstdio>
#include <fstream>
#include <string>

#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#endif

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("system %-60s -> %s\n", what.c_str(), ok ? "ok" : "FAILED   <-- WRONG");
    failures |= !ok;
}
}  // namespace

int main() {
    using namespace basalt::platform;
    check(process_id() > 0, "process_id is positive");
    (void) stderr_is_terminal();   // either answer is valid; the call must not fail

    const std::string mi = meminfo_path();
    if (!mi.empty()) {
        std::ifstream f(mi);
        check((bool) f, "the memory accounting file named is readable");
        check(!self_cgroup_path().empty() && !cgroup_root().empty(), "and the cgroup paths come with it");
    } else {
        check(self_cgroup_path().empty() && cgroup_root().empty(), "no accounting file means no cgroup paths either");
    }
    check(available_memory() > 0, "available_memory reads");

    const ProcessMemory pm = process_memory();
    if (!mi.empty()) check(pm.rss_mib > 0 && pm.avail_mib > 0, "process_memory reads this process's resident set");

    std::string dev;
    const int rot = disk_rotational(temp_dir(), dev);
    check(rot >= -1 && rot <= 1, "disk_rotational answers 1, 0 or unknown");
    const ThreadWaits w = threads_in_disk_wait();
    check(w.waiting <= w.threads, "threads_in_disk_wait counts waiting within all");
    if (!mi.empty()) check(w.threads >= 1, "and lists this process's threads");

    // set_env keeps a value that is already there, so each name is set once
    set_env("BASALT_FORK_SYSTEM_TEST_OFF", "0");
    set_env("BASALT_FORK_SYSTEM_TEST_ON", "1");
    check(!fork_switch("SYSTEM_TEST_OFF"), "fork_switch is off when the variable is 0");
    check(fork_switch("SYSTEM_TEST_ON"), "and on when it is 1");
    check(fork_switch("SYSTEM_TEST_UNSET"), "and on when it is not set");
    set_env("BASALT_FORK_SYSTEM_TEST_NUM", "0.8");
    set_env("BASALT_FORK_SYSTEM_TEST_BIG", "9");
    set_env("BASALT_FORK_SYSTEM_TEST_WORD", "abc");
    check(fork_number("SYSTEM_TEST_NUM", 1.0, 0.25, 4.0) == 0.8, "fork_number reads a number within its range");
    check(fork_number("SYSTEM_TEST_BIG", 1.0, 0.25, 4.0) == 1.0, "and keeps the default outside it");
    check(fork_number("SYSTEM_TEST_WORD", 1.0, 0.25, 4.0) == 1.0, "and for a word");
    check(fork_number("SYSTEM_TEST_UNSET_NUM", 1.0, 0.25, 4.0) == 1.0, "and when it is not set");

    unsigned os_error = 0;
    const FullSpeed fs = keep_full_speed(&os_error);
    check(fs == FullSpeed::NotNeeded || fs == FullSpeed::Set || (fs == FullSpeed::Disabled && !fork_switch("ECOQOS")),
          "keep_full_speed is accepted where the OS slows background processes, and not needed elsewhere");
    keep_thread_full_speed();   // best effort; the call must not fail
#if defined(_WIN32)
    if (fs == FullSpeed::Set) {
        PROCESS_POWER_THROTTLING_STATE st{};
        st.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
        const bool got = GetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &st, sizeof st) != 0;
        check(got && (st.ControlMask & PROCESS_POWER_THROTTLING_EXECUTION_SPEED) != 0 &&
                  (st.StateMask & PROCESS_POWER_THROTTLING_EXECUTION_SPEED) == 0,
              "and the OS reads it back as controlled by the process and never lowered");
    }
#endif

    (void) gpu_peer_access();
    const DynLib nvml = DynLib::open_nvml();
    if (nvml) {
        check(nvml.sym<void*>("nvmlInit_v2") != nullptr, "the management library, when present, exports nvmlInit_v2");
        check(nvml.sym<void*>("basalt_no_such_symbol") == nullptr, "and a missing symbol is null");
    } else {
        std::printf("system: no GPU management library here; the loader returned empty\n");
    }
    DynLib moved = DynLib();
    check(!moved, "an empty DynLib is false");
    return failures ? 1 : 0;
}
