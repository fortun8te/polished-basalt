// tests/vision/hardening_vision_mmproj.cpp - a corrupt projector file is refused with the check that names it.
//
// Every case is a synthetic GGUF built here: one valid file, then the same file with one field made hostile (an
// alignment, a dimension, an offset, a count, a nested array, a repeated tensor).  A case passes only when the open
// fails with the expected message, so a refusal that comes from some other check does not count; a prefix sweep adds
// that no truncation of the valid file is read out of bounds (run it under a sanitizer to see that).
#include "basalt/vision/mmproj.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<uint8_t>;

template <class T>
void put(Bytes& b, T v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    b.insert(b.end(), p, p + sizeof v);
}
void str(Bytes& b, const std::string& s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}
void text(Bytes& b, const char* key, const char* value) {
    str(b, key);
    put<uint32_t>(b, 8);
    str(b, value);
}
// a whole number is stored as u32 like the converter does; anything else as f64
void num(Bytes& b, const char* key, double value) {
    str(b, key);
    if (value >= 0 && value < 4294967296.0 && value == (double) (uint32_t) value) {
        put<uint32_t>(b, 4);
        put<uint32_t>(b, (uint32_t) value);
    } else {
        put<uint32_t>(b, 12);
        put<double>(b, value);
    }
}
void triple(Bytes& b, const char* key) {
    str(b, key);
    put<uint32_t>(b, 9);
    put<uint32_t>(b, 6);
    put<uint64_t>(b, 3);
    for (int i = 0; i < 3; ++i) put<float>(b, 0.5f);
}

struct Tensor {
    std::string name = "t";
    std::vector<uint64_t> dims{2, 2};   // f32, so 16 bytes
    uint64_t off = 0;
};

struct Spec {
    double alignment = 32;              // 0 = leave the key out
    double patch = 16;
    uint64_t kv_count_extra = 0;        // added to the declared metadata count
    uint64_t tensor_count_extra = 0;
    std::vector<Tensor> tensors{Tensor{}};
    Bytes extra_kv;                     // raw entries after the standard ones, counted as one
    size_t data0 = 0;                   // out: where the tensor data starts
};

Bytes fixture(Spec& s) {
    Bytes b;
    put<uint32_t>(b, 0x46554747);
    put<uint32_t>(b, 3);
    put<uint64_t>(b, s.tensors.size() + s.tensor_count_extra);
    put<uint64_t>(b, 12 + (s.alignment != 0) + (s.extra_kv.empty() ? 0 : 1) + s.kv_count_extra);
    text(b, "general.architecture", "clip");
    text(b, "clip.projector_type", "qwen3vl_merger");
    num(b, "clip.vision.image_size", 384);
    num(b, "clip.vision.patch_size", s.patch);
    num(b, "clip.vision.embedding_length", 1152);
    num(b, "clip.vision.feed_forward_length", 4304);
    num(b, "clip.vision.block_count", 27);
    num(b, "clip.vision.attention.head_count", 16);
    num(b, "clip.vision.spatial_merge_size", 2);
    num(b, "clip.vision.projection_dim", 4096);
    if (s.alignment != 0) num(b, "general.alignment", s.alignment);
    triple(b, "clip.vision.image_mean");
    triple(b, "clip.vision.image_std");
    b.insert(b.end(), s.extra_kv.begin(), s.extra_kv.end());
    for (const Tensor& t : s.tensors) {
        str(b, t.name);
        put<uint32_t>(b, (uint32_t) t.dims.size());
        for (uint64_t d : t.dims) put<uint64_t>(b, d);
        put<uint32_t>(b, 0);
        put<uint64_t>(b, t.off);
    }
    const bool pow2 = s.alignment >= 1 && s.alignment <= 1048576 && ((uint64_t) s.alignment & ((uint64_t) s.alignment - 1)) == 0;
    const size_t align = s.alignment == 0 ? 32 : pow2 ? (size_t) s.alignment : 32;
    while (b.size() % align) b.push_back(0);
    s.data0 = b.size();
    b.resize(b.size() + 64);
    return b;
}

// a chain of arrays, each holding one array: read recursively, it takes a stack frame per 12 bytes of file
Bytes nested_arrays(uint32_t depth) {
    Bytes b;
    str(b, "clip.nested");
    put<uint32_t>(b, 9);
    for (uint32_t i = 0; i < depth; ++i) {
        put<uint32_t>(b, 9);
        put<uint64_t>(b, 1);
    }
    put<uint32_t>(b, 6);
    put<uint64_t>(b, 0);
    return b;
}

int failures = 0;
const std::filesystem::path path = std::filesystem::temp_directory_path() / "basalt-vision-mmproj-hardening.gguf";

// opens `bytes`; expect == nullptr means it must open, otherwise it must fail with a message containing expect
void check(const char* name, const Bytes& bytes, const char* expect) {
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(bytes.data()), (std::streamsize) bytes.size());
    }
    std::string err;
    bool ok;
    {
        basalt::vision::Mmproj mp;
        ok = mp.open(path.string(), err);
    }
    const bool good = expect == nullptr ? ok && err.empty() : !ok && err.find(expect) != std::string::npos;
    if (!good) {
        std::printf("FAIL %s: open %s, message '%s', wanted %s\n", name, ok ? "succeeded" : "failed", err.c_str(),
                    expect == nullptr ? "success" : expect);
        ++failures;
    }
}

}  // namespace

int main() {
    const uint64_t kInt64Max = (uint64_t) std::numeric_limits<int64_t>::max();
    Spec valid;
    const Bytes good = fixture(valid);
    check("valid", good, nullptr);

    auto spec = [](auto edit) { Spec s; edit(s); return s; };
    auto run = [&](const char* name, Spec s, const char* expect) { const Bytes b = fixture(s); check(name, b, expect); };

    run("alignment 64", spec([](Spec& s) { s.alignment = 64; }), nullptr);
    run("alignment 1", spec([](Spec& s) { s.alignment = 1; }), nullptr);
    run("alignment left out", spec([](Spec& s) { s.alignment = 0; }), nullptr);
    run("alignment negative", spec([](Spec& s) { s.alignment = -32; }), "invalid GGUF alignment");
    run("alignment not a power of two", spec([](Spec& s) { s.alignment = 3; }), "invalid GGUF alignment");
    run("alignment fractional", spec([](Spec& s) { s.alignment = 32.5; }), "invalid GGUF alignment");
    run("alignment past the cap", spec([](Spec& s) { s.alignment = 4294967296.0; }), "invalid GGUF alignment");

    run("patch size zero", spec([](Spec& s) { s.patch = 0; }), "invalid metadata clip.vision.patch_size");
    run("patch size fractional", spec([](Spec& s) { s.patch = 16.5; }), "invalid metadata clip.vision.patch_size");
    run("patch size past int", spec([](Spec& s) { s.patch = 4294967296.0; }), "invalid metadata clip.vision.patch_size");

    run("dimension zero", spec([](Spec& s) { s.tensors[0].dims = {0, 2}; }), "invalid tensor dimension");
    run("dimension past int64", spec([](Spec& s) { s.tensors[0].dims = {kInt64Max + 1, 2}; }), "invalid tensor dimension");
    run("rank zero", spec([](Spec& s) { s.tensors[0].dims = {}; }), "bad tensor rank");
    run("rank five", spec([](Spec& s) { s.tensors[0].dims = {1, 1, 1, 1, 1}; }), "bad tensor rank");
    // 2^32 x 2^32 wraps a 64-bit product to 0; INT64_MAX x 2 overflows it
    run("elements wrap to zero", spec([](Spec& s) { s.tensors[0].dims = {1ull << 32, 1ull << 32}; }),
        "tensor dimensions overflow");
    run("elements overflow", spec([](Spec& s) { s.tensors[0].dims = {kInt64Max, 2}; }), "tensor dimensions overflow");
    run("bytes overflow", spec([](Spec& s) { s.tensors[0].dims = {kInt64Max}; }), "tensor size overflow");
    run("tensor larger than the file", spec([](Spec& s) { s.tensors[0].dims = {1u << 20, 1u << 10}; }),
        "runs past the end");

    // data0 + off + bytes wraps back inside the file for these
    run("offset wraps the address", spec([](Spec& s) { s.tensors[0].off = std::numeric_limits<uint64_t>::max() - 7; }),
        "runs past the end");
    run("offset past the end", spec([](Spec& s) { s.tensors[0].off = 1u << 20; }), "runs past the end");
    run("offset at the last byte", spec([](Spec& s) { s.tensors[0].off = 64 - 15; }), "runs past the end");
    run("offset at the last fit", spec([](Spec& s) { s.tensors[0].off = 64 - 16; }), nullptr);

    run("duplicate tensor name", spec([](Spec& s) { s.tensors = {Tensor{}, Tensor{}}; }), "duplicate tensor name t");
    run("two tensors", spec([](Spec& s) { s.tensors = {Tensor{}, Tensor{"u", {2, 2}, 16}}; }), nullptr);

    run("tensor count past the file", spec([](Spec& s) { s.tensor_count_extra = 1ull << 40; }), "invalid GGUF entry counts");
    run("metadata count past the file", spec([](Spec& s) { s.kv_count_extra = 1ull << 40; }), "invalid GGUF entry counts");
    // the next entry is read from the tensor directory: any refusal will do, it must not be read out of bounds
    run("metadata count above the entries", spec([](Spec& s) { s.kv_count_extra = 1; }), "");

    run("array of arrays", spec([](Spec& s) { s.extra_kv = nested_arrays(2); }), "truncated GGUF metadata");
    // 400000 levels is 4.8 MB of file and well past any thread's stack when read recursively
    run("deeply nested arrays", spec([](Spec& s) { s.extra_kv = nested_arrays(400000); }), "truncated GGUF metadata");
    {
        Bytes huge;                     // an array that claims 2^62 floats
        str(huge, "clip.huge");
        put<uint32_t>(huge, 9);
        put<uint32_t>(huge, 6);
        put<uint64_t>(huge, 1ull << 62);
        run("array count past the file", spec([&](Spec& s) { s.extra_kv = huge; }), "truncated GGUF metadata");
        Bytes unknown;                  // an array of an element type that does not exist
        str(unknown, "clip.unknown");
        put<uint32_t>(unknown, 9);
        put<uint32_t>(unknown, 99);
        put<uint64_t>(unknown, 1);
        run("array of an unknown type", spec([&](Spec& s) { s.extra_kv = unknown; }), "truncated GGUF metadata");
    }

    // no prefix of the valid file is read out of bounds, and none that cuts the tensor data is accepted
    for (size_t len = 0; len < good.size(); ++len) {
        const Bytes cut(good.begin(), good.begin() + (std::ptrdiff_t) len);
        const bool whole = len >= valid.data0 + 16;
        char name[48];
        std::snprintf(name, sizeof name, "prefix %zu", len);
        if (whole) continue;            // only the unused tail is missing: it may open
        check(name, cut, len < 32 ? "cannot read" : "");
        if (failures > 3) break;
    }
    std::filesystem::remove(path);
    if (failures != 0) return 1;
    std::printf("mmproj hardening: ok\n");
    return 0;
}
