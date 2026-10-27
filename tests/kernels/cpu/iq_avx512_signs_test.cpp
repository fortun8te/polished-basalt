// tests/kernels/cpu/iq_avx512_signs_test.cpp - the AVX-512 i-quant rows with sign masks from the BMI2 deposit against
// the same rows with the ksigns table, bitwise.
//
//     build/iq_avx512_signs_test
//
// Gate/up rows and plain dot rows of IQ2_XXS, IQ2_XS and IQ3_XXS (the formats that store sign indices), plus IQ3_S and
// IQ2_S as controls, on random weights and on weights with every code at its extreme, with the model's row length,
// 1 to 8 tokens. The two builds must agree to the bit on every output. Exits 77 (skipped) on a CPU that does not run
// the AVX-512 kernels. The table-free construction itself is checked exhaustively by iq_signs_test.
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/iq_avx512.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace cpu = basalt::kernels::cpu;

namespace {

constexpr int N = cpu::H, ROWS = 48, MAXT = cpu::MAXT;

// Random bytes with a valid small fp16 scale at the start of each block; mode 1 pins every code to its extreme.
void fill(std::vector<uint8_t>& v, int type, uint64_t seed, int mode) {
    std::mt19937_64 rng(seed);
    for (size_t i = 0; i + 8 <= v.size(); i += 8) {
        const uint64_t x = mode == 1 ? ~0ull : rng();
        std::memcpy(v.data() + i, &x, 8);
    }
    const size_t bs = ggml_type_size((ggml_type) type);
    for (size_t b = 0; b + bs <= v.size(); b += bs) {
        const uint16_t d = (uint16_t) (0x2800 + (v[b] & 0xff));
        std::memcpy(&v[b], &d, 2);
    }
}

int failures = 0;

bool same_outputs(const std::vector<float>& a, const std::vector<float>& b, size_t written) {
    return std::memcmp(a.data(), b.data(), written * sizeof(float)) == 0;
}

bool tail_intact(const std::vector<float>& values, size_t written, float sentinel) {
    for (size_t i = written; i < values.size(); ++i)
        if (std::memcmp(&values[i], &sentinel, sizeof(float)) != 0) return false;
    return true;
}

void run(int type, int mode, uint64_t seed) {
    const size_t row = ggml_row_size((ggml_type) type, N);
    std::vector<uint8_t> w(2 * ROWS * row);
    fill(w, type, seed, mode);
    // activations: Q8_K blocks of random floats (ggml's block_q8_K, one per 256 values)
    const size_t act_bytes = ggml_row_size(GGML_TYPE_Q8_K, N);
    std::vector<std::vector<uint8_t>> act(MAXT, std::vector<uint8_t>(act_bytes));
    std::mt19937 rng((uint32_t) seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(N);
    for (int t = 0; t < MAXT; ++t) {
        for (float& f : x) f = nd(rng);
        cpu::quantize_q8_K_avx512(x.data(), act[t].data(), N);
    }
    const void* a[MAXT];
    for (int t = 0; t < MAXT; ++t) a[t] = act[t].data();

    for (int nt = 1; nt <= MAXT; ++nt) {
        std::vector<float> ff0(MAXT * ROWS, -1.f), ff1(MAXT * ROWS, -2.f), o0(MAXT * ROWS, -1.f), o1(MAXT * ROWS, -2.f);
        float* f0[MAXT]; float* f1[MAXT]; float* p0[MAXT]; float* p1[MAXT];
        for (int t = 0; t < MAXT; ++t) { f0[t] = &ff0[t * ROWS]; f1[t] = &ff1[t * ROWS]; p0[t] = &o0[t * ROWS]; p1[t] = &o1[t * ROWS]; }
        cpu::iq512_gu_rows_signs(false, type, w.data(), row, ROWS * row, N, a, nt, f0, 0, ROWS);
        cpu::iq512_gu_rows_signs(true, type, w.data(), row, ROWS * row, N, a, nt, f1, 0, ROWS);
        cpu::iq512_rows_signs(false, type, w.data(), row, N, a, nt, p0, 0, ROWS);
        cpu::iq512_rows_signs(true, type, w.data(), row, N, a, nt, p1, 0, ROWS);
        const size_t written = (size_t) nt * ROWS;
        const bool same_gu = same_outputs(ff0, ff1, written);
        const bool same_dot = same_outputs(o0, o1, written);
        const bool tails_ok = tail_intact(ff0, written, -1.f) && tail_intact(ff1, written, -2.f) &&
                              tail_intact(o0, written, -1.f) && tail_intact(o1, written, -2.f);
        if (!same_gu || !same_dot || !tails_ok) {
            std::printf("type %d mode %d nt %d: gate/up %s, dot %s, unused tails %s   <-- WRONG\n",
                        type, mode, nt, same_gu ? "same" : "DIFFERENT", same_dot ? "same" : "DIFFERENT",
                        tails_ok ? "intact" : "CLOBBERED");
            failures = 1;
        }
    }
}

}  // namespace

int main() {
    if (!cpu::cpu_avx512_ok()) {
        std::printf("iq_avx512_signs: this CPU does not run the AVX-512 kernels; skipped\n");
        return 77;
    }
    std::printf("iq_avx512_signs: process choice = %s\n", cpu::cpu_pdep_signs_ok() ? "pdep" : "table");
    const int types[] = {16, 17, 18, 21, 22};
    for (int type : types)
        for (int mode = 0; mode < 2; ++mode)
            for (uint64_t seed = 1; seed <= (mode ? 1u : 6u); ++seed) run(type, mode, seed * 7919 + (uint64_t) type);
    std::printf("iq_avx512_signs: %s\n", failures ? "FAILED" : "ok");
    return failures;
}
