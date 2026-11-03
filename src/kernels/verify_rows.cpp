// src/kernels/verify_rows.cpp - the runtime cap on a verify window's rows (see verify_kernels.hpp).
#include "basalt/kernels/verify_kernels.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>

namespace basalt::kernels {
namespace {
// 0: not set yet (BASALT_VERIFY_MAX_ROWS, else kVerifyMaxT)
std::atomic<int> g_rows{0};
int clamp_rows(long v) { return (int) std::min<long>(std::max<long>(v, 2), kVerifyMaxT); }
}  // namespace

int verify_max_rows() {
    int r = g_rows.load(std::memory_order_acquire);
    if (r == 0) {
        const char* e = std::getenv("BASALT_VERIFY_MAX_ROWS");
        r = e != nullptr && *e != '\0' ? clamp_rows(std::atol(e)) : kVerifyMaxT;
        g_rows.store(r, std::memory_order_release);
    }
    return r;
}

void set_verify_max_rows(int rows) { g_rows.store(clamp_rows(rows), std::memory_order_release); }

}  // namespace basalt::kernels
