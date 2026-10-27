// include/basalt/kernels/cpu/iq_signs.hpp - the sign bytes of the i-quant formats without the ksigns table.
//
// IQ2_XXS, IQ2_XS and IQ3_XXS keep each group's signs as a 7-bit index into ggml's `ksigns_iq2xs`. An entry is the
// index with an eighth bit that makes the byte's popcount even: ksigns_iq2xs[i] == i | parity(i) << 7. Eight indices
// side by side in a word, one per byte, turn into the eight sign bytes with three xor-shifts. The parallel bit
// deposit and extract that put the indices there are passed in, so the same code runs on the BMI2 instructions and,
// in the tests, on a scalar emulation.
#pragma once

#include <cstdint>

namespace basalt::kernels::cpu {

/// `idx` holds eight 7-bit indices, one in the low seven bits of each byte (bit 7 of every byte clear). Returns the
/// eight bytes ksigns_iq2xs[idx_l], in the same byte order. Each fold only has to be right in bit 0..3 of the byte
/// that is read back: after the folds by 4, 2 and 1 bit 0 of a byte is the xor of that byte's own eight bits, and the
/// bits that cross into it from the next byte stay in bits 4-7, which the later folds do not read into bit 0.
inline uint64_t iq_sign_bytes(uint64_t idx) {
    uint64_t p = idx ^ (idx >> 4);
    p ^= p >> 2;
    p ^= p >> 1;
    return idx | ((p & 0x0101010101010101ull) << 7);
}

/// `groups` holds eight 7-bit indices back to back (index l in bits 7l..7l+6; bits above 55 are not read).
/// `deposit(src, mask)` is the BMI2 parallel bit deposit (`_pdep_u64`).
template <class Deposit>
inline uint64_t iq_sign_bytes_packed(uint64_t groups, Deposit deposit) {
    return iq_sign_bytes(deposit(groups, 0x7F7F7F7F7F7F7F7Full));
}

/// IQ2_XXS and IQ3_XXS: each of two u32 words holds four 7-bit indices in bits 0-27 and a scale in bits 28-31.
/// Returns the eight indices back to back.
inline uint64_t iq_xxs_index_groups(uint32_t a, uint32_t c) {
    return (uint64_t) (a & 0x0FFFFFFFu) | ((uint64_t) c << 28);
}

/// IQ2_XS: eight u16 entries, as two u64 words of four, keep the 7-bit index in bits 9-15 of each. Returns the eight
/// indices back to back. `extract(src, mask)` is the BMI2 parallel bit extract (`_pext_u64`).
template <class Extract>
inline uint64_t iq2xs_index_groups(uint64_t w0, uint64_t w1, Extract extract) {
    constexpr uint64_t top7 = 0xFE00FE00FE00FE00ull;
    return extract(w0, top7) | (extract(w1, top7) << 28);
}

}  // namespace basalt::kernels::cpu
