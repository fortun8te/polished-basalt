// tests/kernels/cpu/iq_signs_test.cpp - `iq_signs.hpp` against ggml's ksigns_iq2xs table (CPU only; runs on any host).
//
//     build/iq_signs_test [--exhaustive]
//
// The AVX-512 i-quant rows build their sign masks from a parallel bit deposit and a parity fold instead of eight table
// lookups. Both must give the same eight bytes for every input. The deposit and extract are a scalar emulation here
// (and the hardware instruction too where this build has it), so the test needs no BMI2. Checks:
//
//   1. ksigns_iq2xs[i] == i | parity(i) << 7 for all 128 indices, parity counted bit by bit;
//   2. the emulation of deposit and extract on known values, and each against the other;
//   3. the byte fold alone, against the table: every index in every lane, every ordered pair in every pair of
//      neighbouring lanes (the only place bits cross), every triple in six runs of three lanes;
//   4. the IQ2_XXS / IQ3_XXS words (four indices and a scale in each of two u32) against the kernel's table lookups,
//      on random words, and with --exhaustive on all 2^28 index sets of each word against several fixed partners;
//   5. the IQ2_XS entries (eight u16, index in bits 9-15) the same way, with every 16-bit value in every lane.
//
// Eight independent lanes of 128 values cannot be enumerated together (2^56); step 3 exercises each lane and every
// pair of neighbours exhaustively and the later steps check the packing around them.
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "basalt/ggml/ggml_common.h"
#include "basalt/kernels/cpu/iq_signs.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#if defined(__BMI2__)
#include <immintrin.h>
#endif

namespace cpu = basalt::kernels::cpu;

namespace {

std::atomic<unsigned long long> bad{0}, checked{0};
thread_local unsigned long long local_checked = 0;   // folded into `checked` when a thread ends: no shared counter per compare

// pdep: the low bits of `src` go, in order, to the set bits of `mask` (lowest first)
uint64_t pdep_emu(uint64_t src, uint64_t mask) {
    uint64_t r = 0;
    for (uint64_t m = mask, k = 1; m; m &= m - 1, k <<= 1)
        if (src & k) r |= m & (~m + 1);
    return r;
}
// pext: the bits of `src` at the set bits of `mask` are gathered, in order, into the low bits
uint64_t pext_emu(uint64_t src, uint64_t mask) {
    uint64_t r = 0;
    for (uint64_t m = mask, k = 1; m; m &= m - 1, k <<= 1)
        if (src & (m & (~m + 1))) r |= k;
    return r;
}

int popcount64(uint64_t m) {
    int n = 0;
    for (; m; m &= m - 1) ++n;
    return n;
}

int parity7(int i) {
    int p = 0;
    for (int b = 0; b < 8; ++b) p ^= (i >> b) & 1;
    return p;
}

// ---- the kernels' table forms, as they were

uint64_t table_bytes(uint64_t idx) {   // eight index bytes -> eight sign bytes
    uint64_t s = 0;
    for (int l = 0; l < 8; ++l) s |= (uint64_t) ksigns_iq2xs[(idx >> (8 * l)) & 127] << (8 * l);
    return s;
}
uint64_t table_xxs(uint32_t a, uint32_t c) {   // IQ2_XXS / IQ3_XXS
    return (uint64_t) ksigns_iq2xs[a & 127] | ((uint64_t) ksigns_iq2xs[(a >> 7) & 127] << 8) |
           ((uint64_t) ksigns_iq2xs[(a >> 14) & 127] << 16) | ((uint64_t) ksigns_iq2xs[(a >> 21) & 127] << 24) |
           ((uint64_t) ksigns_iq2xs[c & 127] << 32) | ((uint64_t) ksigns_iq2xs[(c >> 7) & 127] << 40) |
           ((uint64_t) ksigns_iq2xs[(c >> 14) & 127] << 48) | ((uint64_t) ksigns_iq2xs[(c >> 21) & 127] << 56);
}
uint64_t table_xs(const uint16_t* v) {   // IQ2_XS
    uint64_t s = 0;
    for (int l = 0; l < 8; ++l) s |= (uint64_t) ksigns_iq2xs[v[l] >> 9] << (8 * l);
    return s;
}

// ---- the new forms on the emulation

struct Dep { uint64_t operator()(uint64_t s, uint64_t m) const { return pdep_emu(s, m); } };
struct Ext { uint64_t operator()(uint64_t s, uint64_t m) const { return pext_emu(s, m); } };

uint64_t new_xxs(uint32_t a, uint32_t c) { return cpu::iq_sign_bytes_packed(cpu::iq_xxs_index_groups(a, c), Dep{}); }
uint64_t new_xs(const uint16_t* v) {
    uint64_t w0, w1;
    std::memcpy(&w0, v, 8);
    std::memcpy(&w1, v + 4, 8);
    return cpu::iq_sign_bytes_packed(cpu::iq2xs_index_groups(w0, w1, Ext{}), Dep{});
}

void expect(uint64_t got, uint64_t want, const char* what, unsigned long long input) {
    ++local_checked;
    if (got != want && bad++ < 8)
        std::printf("MISMATCH %s input=%016llx got=%016llx want=%016llx\n", what, input, (unsigned long long) got,
                    (unsigned long long) want);
}

template <class F>
void parallel_for(uint64_t n, F f) {
    const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nt; ++t)
        th.emplace_back([=] {
            for (uint64_t i = t; i < n; i += nt) f(i);
            checked += local_checked;
        });
    for (auto& x : th) x.join();
}

}  // namespace

int main(int argc, char** argv) {
    const bool exhaustive = argc > 1 && std::strcmp(argv[1], "--exhaustive") == 0;

    // 1. the table is the rule
    for (int i = 0; i < 128; ++i) expect(ksigns_iq2xs[i], (uint64_t) (i | (parity7(i) << 7)), "ksigns rule", (unsigned) i);
    std::printf("1. ksigns_iq2xs[i] == i | parity(i) << 7 for all 128 indices: %s\n", bad ? "FAILED" : "ok");

    // 2. the emulation
    expect(pdep_emu(0b101, 0b11010), 0b10010, "pdep known", 0);
    expect(pext_emu(0b10010, 0b11010), 0b101, "pext known", 0);
    expect(pdep_emu(~0ull, 0x7F7F7F7F7F7F7F7Full), 0x7F7F7F7F7F7F7F7Full, "pdep full", 0);
    expect(pext_emu(0xFE00FE00FE00FE00ull, 0xFE00FE00FE00FE00ull), 0x0FFFFFFFull, "pext full", 0);
    std::mt19937_64 rng(0x5eed);
    for (int i = 0; i < 100000; ++i) {
        const uint64_t x = rng(), m = rng() & rng();
        const int pc = popcount64(m);
        const uint64_t low = pc == 64 ? ~0ull : ((1ull << pc) - 1);
        expect(pext_emu(pdep_emu(x, m), m), x & low, "pext(pdep)", x);
        expect(pdep_emu(pext_emu(x, m), m), x & m, "pdep(pext)", x);
#if defined(__BMI2__)
        expect(_pdep_u64(x, m), pdep_emu(x, m), "hardware pdep", x);
        expect(_pext_u64(x, m), pext_emu(x, m), "hardware pext", x);
#endif
    }
    std::printf("2. deposit/extract emulation%s: %s\n",
#if defined(__BMI2__)
                " (and the hardware instructions)",
#else
                "",
#endif
                bad ? "FAILED" : "ok");

    // 3. the byte fold: bytes hold 7-bit values
    const unsigned long long before3 = bad;
    for (int l = 0; l < 8; ++l)
        for (uint64_t v = 0; v < 128; ++v) expect(cpu::iq_sign_bytes(v << (8 * l)), table_bytes(v << (8 * l)), "single", v);
    for (int l = 0; l < 7; ++l)
        for (uint64_t u = 0; u < 128; ++u)
            for (uint64_t v = 0; v < 128; ++v) {
                uint64_t rest = rng() & 0x7F7F7F7F7F7F7F7Full;
                rest &= ~(0xFFFFull << (8 * l));
                const uint64_t x = rest | (u << (8 * l)) | (v << (8 * l + 8));
                expect(cpu::iq_sign_bytes(x), table_bytes(x), "pair", x);
            }
    for (int l = 0; l < 6; ++l)
        parallel_for(128 * 128, [&](uint64_t uv) {
            for (uint64_t w = 0; w < 128; ++w) {
                const uint64_t x = ((uv & 127) << (8 * l)) | ((uv >> 7) << (8 * l + 8)) | (w << (8 * l + 16));
                expect(cpu::iq_sign_bytes(x), table_bytes(x), "triple", x);
            }
        });
    for (int i = 0; i < 4000000; ++i) {
        const uint64_t x = rng() & 0x7F7F7F7F7F7F7F7Full;
        expect(cpu::iq_sign_bytes(x), table_bytes(x), "random 8 lanes", x);
    }
    std::printf("3. byte fold: every index per lane, every neighbouring pair, every triple, 4M random words: %s\n",
                bad == before3 ? "ok" : "FAILED");

    // 4. IQ2_XXS / IQ3_XXS words
    const unsigned long long before4 = bad;
    for (int i = 0; i < 4000000; ++i) {
        const uint32_t a = (uint32_t) rng(), c = (uint32_t) rng();
        expect(new_xxs(a, c), table_xxs(a, c), "xxs random", ((uint64_t) a << 32) | c);
    }
    const uint64_t n28 = exhaustive ? (1ull << 28) : (1ull << 20);
    const uint32_t stride = exhaustive ? 1 : 256;   // without --exhaustive: every 256th set of indices
    const uint32_t partners[] = {0u, 0xFFFFFFFFu, 0x0FFFFFFFu, 0xF5555555u, 0x0AAAAAAAu, 0x12345678u};
    for (uint32_t p : partners) {
        parallel_for(n28, [&](uint64_t i) {
            const uint32_t idx = (uint32_t) i * stride, hi = (uint32_t) (idx * 0x9E3779B1u) & 0xF0000000u;
            expect(new_xxs(idx | hi, p), table_xxs(idx | hi, p), "xxs first word", ((uint64_t) (idx | hi) << 32) | p);
            expect(new_xxs(p, idx | hi), table_xxs(p, idx | hi), "xxs second word", ((uint64_t) p << 32) | (idx | hi));
        });
    }
    std::printf("4. XXS words (%s): %s\n", exhaustive ? "all 2^28 index sets of each word x 6 partners + 4M random"
                                                       : "sampled; --exhaustive runs every set", bad == before4 ? "ok" : "FAILED");

    // 5. IQ2_XS entries
    const unsigned long long before5 = bad;
    for (int l = 0; l < 8; ++l)
        for (uint32_t e = 0; e < 65536; ++e) {
            uint16_t v[8];
            for (int k = 0; k < 8; ++k) v[k] = (uint16_t) rng();
            v[l] = (uint16_t) e;
            expect(new_xs(v), table_xs(v), "xs lane", ((uint64_t) l << 32) | e);
        }
    for (int i = 0; i < 4000000; ++i) {
        uint16_t v[8];
        for (int k = 0; k < 8; ++k) v[k] = (uint16_t) rng();
        expect(new_xs(v), table_xs(v), "xs random", v[0]);
    }
    std::printf("5. XS entries: every 16-bit value in every lane, 4M random sets: %s\n", bad == before5 ? "ok" : "FAILED");

    checked += local_checked;
    std::printf("iq_signs: %llu comparisons, %llu mismatches -> %s\n", (unsigned long long) checked,
                (unsigned long long) bad, bad ? "FAILED" : "ok");
    return bad ? 1 : 0;
}
