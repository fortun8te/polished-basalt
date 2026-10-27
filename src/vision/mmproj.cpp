// src/vision/mmproj.cpp - see basalt/vision/mmproj.hpp.
#include "basalt/vision/mmproj.hpp"

#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace basalt::vision {

namespace {

struct Cursor {
    const uint8_t* p;
    const uint8_t* end;
    bool ok = true;
    template <class T>
    T get() {
        T v{};
        if ((size_t) (end - p) < sizeof(T)) { ok = false; return v; }
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        const uint64_t n = get<uint64_t>();
        if (!ok || (uint64_t) (end - p) < n) { ok = false; return {}; }
        std::string s((const char*) p, n);
        p += n;
        return s;
    }
};

size_t scalar_size(uint32_t t) {
    switch (t) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
        default: return 0;
    }
}

// A metadata value: numbers are kept as doubles (one array of numbers as well); strings and other arrays skipped.
struct Value {
    bool num = false;
    double d = 0;
    std::vector<double> arr;
};

Value read_value(Cursor& c, uint32_t t) {
    Value v;
    if (t == 8) { c.str(); return v; }
    if (t == 9) {
        const uint32_t et = c.get<uint32_t>();
        const uint64_t n = c.get<uint64_t>();
        // Elements are scalars or strings, never arrays (a chain of nested arrays would recurse once per 12 bytes of
        // file), and each takes at least its size in the file, so a count the rest of the file cannot hold is refused.
        const size_t es = et == 8 ? sizeof(uint64_t) : scalar_size(et);
        if (!c.ok || et == 9 || es == 0 || n > (uint64_t) (c.end - c.p) / es) { c.ok = false; return v; }
        for (uint64_t i = 0; c.ok && i < n; ++i) {
            const Value e = read_value(c, et);
            if (e.num && n <= 4096) v.arr.push_back(e.d);
        }
        return v;
    }
    v.num = true;
    switch (t) {
        case 0: v.d = c.get<uint8_t>(); break;
        case 1: v.d = c.get<int8_t>(); break;
        case 2: v.d = c.get<uint16_t>(); break;
        case 3: v.d = c.get<int16_t>(); break;
        case 4: v.d = c.get<uint32_t>(); break;
        case 5: v.d = c.get<int32_t>(); break;
        case 6: v.d = c.get<float>(); break;
        case 7: v.d = c.get<uint8_t>(); break;
        case 10: v.d = (double) c.get<uint64_t>(); break;
        case 11: v.d = (double) c.get<int64_t>(); break;
        case 12: v.d = c.get<double>(); break;
        default: c.ok = false;
    }
    return v;
}

}  // namespace

void Mmproj::release() {
    if (map_ && !mapped_) basalt::platform::free_aligned(map_);
    mapping_.reset();
    mapped_ = false;
    map_ = nullptr;
    tensors_.clear();
}

Mmproj::~Mmproj() { release(); }

bool Mmproj::open(const std::string& path, std::string& err) {
    release();
    err.clear();
    uint64_t file_bytes = 0;
    if (!platform::file_size(path, file_bytes)) { err = "cannot open " + path; return false; }
    if (file_bytes < 32) { err = "cannot read " + path; return false; }
    if (file_bytes > std::numeric_limits<size_t>::max() - 4095 ||
        file_bytes > (uint64_t) std::numeric_limits<ptrdiff_t>::max()) { err = "invalid projector size"; return false; }
    size_ = (size_t) file_bytes;
    // read whole with block-aligned unbuffered reads, so the file cache neither copies nor keeps it; a file system that
    // refuses them, and a platform with no unbuffered descriptor, is mapped instead
    {
        constexpr size_t kAlign = 4096, kChunk = 16u << 20;
        const int dfd = platform::open_unbuffered(path);
        void* mem = nullptr;
        if (dfd >= 0) mem = basalt::platform::alloc_aligned((size_t) ((size_ + kAlign - 1) / kAlign * kAlign), kAlign);
        if (mem != nullptr) {
            size_t got = 0;
            while (got < size_) {
                const size_t want = std::min<size_t>(kChunk, (size_ - got + kAlign - 1) / kAlign * kAlign);
                const int64_t r = basalt::platform::pread(dfd, (uint8_t*) mem + got, want, got);
                if (r <= 0) break;
                got += (size_t) r;
            }
            if (got >= size_) { map_ = mem; mapped_ = false; }
            else basalt::platform::free_aligned(mem);
        }
        if (dfd >= 0) platform::close_fd(dfd);
    }
    if (map_ == nullptr) {
        std::string why;
        auto m = platform::MappedFile::open_ro(path, &why);
        if (!m) { err = "cannot map " + path; return false; }
        mapping_ = std::move(*m);
        map_ = const_cast<uint8_t*>(mapping_.data());
        mapped_ = true;
    }
    const uint8_t* base = (const uint8_t*) map_;
    Cursor c{base, base + size_};
    if (c.get<uint32_t>() != 0x46554747u) { err = path + " is not a GGUF file"; return false; }
    if (c.get<uint32_t>() != 3) { err = "unsupported GGUF version"; return false; }
    const uint64_t n_tensors = c.get<uint64_t>(), n_kv = c.get<uint64_t>();
    // a tensor entry is at least 24 bytes and a metadata entry at least 12, after the 24-byte header
    if (!c.ok || n_tensors > (size_ - 24) / 24 || n_kv > (size_ - 24) / 12) {
        err = "invalid GGUF entry counts"; return false;
    }
    std::map<std::string, Value> kv;
    std::string arch, proj_type;
    uint32_t align = 32;
    for (uint64_t i = 0; c.ok && i < n_kv; ++i) {
        const std::string key = c.str();
        const uint32_t t = c.get<uint32_t>();
        if (t == 8) {
            const std::string s = c.str();
            if (key == "general.architecture") arch = s;
            else if (key == "clip.projector_type") proj_type = s;
            continue;
        }
        kv[key] = read_value(c, t);
    }
    if (!c.ok) { err = "truncated GGUF metadata"; return false; }
    if (arch != "clip" || proj_type != "qwen3vl_merger") {
        err = "not a qwen3vl_merger projector (architecture '" + arch + "', projector '" + proj_type + "')";
        return false;
    }
    auto num = [&](const char* k, double def, bool must = true) {
        auto it = kv.find(k);
        if (it == kv.end() || !it->second.num) {
            if (must) err = std::string("missing metadata ") + k;
            return def;
        }
        return it->second.d;
    };
    // a size, count or step of the encoder: a whole number of at least 1 that fits an int (the first error stays)
    auto integer = [&](const char* k, int def, bool must = true) {
        const double v = num(k, def, must);
        if (!err.empty()) return 0;
        if (!std::isfinite(v) || v < 1 || v > std::numeric_limits<int>::max() || std::floor(v) != v) {
            err = std::string("invalid metadata ") + k;
            return 0;
        }
        return (int) v;
    };
    hp_.image_size = integer("clip.vision.image_size", 0);
    hp_.patch = integer("clip.vision.patch_size", 0);
    hp_.embd = integer("clip.vision.embedding_length", 0);
    hp_.ffn = integer("clip.vision.feed_forward_length", 0);
    hp_.layers = integer("clip.vision.block_count", 0);
    hp_.heads = integer("clip.vision.attention.head_count", 0);
    hp_.merge = integer("clip.vision.spatial_merge_size", 2, false);
    hp_.proj = integer("clip.vision.projection_dim", 0);
    hp_.eps = (float) num("clip.vision.attention.layer_norm_epsilon", 1e-6, false);
    if (!err.empty()) return false;
    for (const char* k : {"clip.vision.image_mean", "clip.vision.image_std"}) {
        auto it = kv.find(k);
        if (it == kv.end() || it->second.arr.size() != 3) { err = std::string("missing metadata ") + k; return false; }
        for (int i = 0; i < 3; ++i) (k[18] == 'm' ? hp_.mean : hp_.std)[i] = (float) it->second.arr[i];
    }
    if (kv.count("general.alignment")) {
        const Value& a = kv["general.alignment"];
        if (!a.num || !std::isfinite(a.d) || a.d < 1 || a.d > 1048576 || std::floor(a.d) != a.d) {
            err = "invalid GGUF alignment"; return false;
        }
        align = (uint32_t) a.d;
    }
    if ((align & (align - 1)) != 0) { err = "invalid GGUF alignment"; return false; }
    struct Info { std::string name; uint32_t type; std::vector<int64_t> ne; uint64_t off; };
    std::vector<Info> infos;
    for (uint64_t i = 0; c.ok && i < n_tensors; ++i) {
        Info t;
        t.name = c.str();
        const uint32_t nd = c.get<uint32_t>();
        if (nd == 0 || nd > 4) { err = "bad tensor rank"; return false; }
        for (uint32_t d = 0; d < nd; ++d) {
            const uint64_t dim = c.get<uint64_t>();
            if (dim == 0 || dim > (uint64_t) std::numeric_limits<int64_t>::max()) {
                err = "invalid tensor dimension"; return false;
            }
            t.ne.push_back((int64_t) dim);
        }
        t.type = c.get<uint32_t>();
        t.off = c.get<uint64_t>();
        infos.push_back(std::move(t));
    }
    if (!c.ok) { err = "truncated GGUF tensor directory"; return false; }
    const size_t hdr = (size_t) (c.p - base);
    if (align - 1 > std::numeric_limits<size_t>::max() - hdr) { err = "invalid GGUF data offset"; return false; }
    const size_t data0 = (hdr + align - 1) / align * align;
    if (data0 > size_) { err = "invalid GGUF data offset"; return false; }
    for (auto& t : infos) {
        TensorRef r;
        r.name = t.name;
        r.type = t.type;
        r.ne = t.ne;
        const size_t esz = t.type == 0 ? 4 : (t.type == 1 || t.type == 30) ? 2 : 0;
        if (esz == 0) { err = "tensor " + t.name + " has an unsupported type " + std::to_string(t.type); return false; }
        int64_t elements = 1;
        for (int64_t d : r.ne) {
            if (elements > std::numeric_limits<int64_t>::max() / d) { err = "tensor dimensions overflow"; return false; }
            elements *= d;
        }
        if ((uint64_t) elements > std::numeric_limits<size_t>::max() / esz) { err = "tensor size overflow"; return false; }
        r.bytes = (size_t) elements * esz;
        if (t.off > size_ - data0 || r.bytes > size_ - data0 - (size_t) t.off) {
            err = "tensor " + t.name + " runs past the end of the file"; return false;
        }
        r.data = base + data0 + (size_t) t.off;
        if (!tensors_.emplace(t.name, std::move(r)).second) { err = "duplicate tensor name " + t.name; return false; }
    }
    return true;
}

const TensorRef* Mmproj::find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

const TensorRef* Mmproj::need(const std::string& name, uint32_t type, int64_t elems, std::string& err) const {
    const TensorRef* t = find(name);
    if (!t) { err = "the projector has no tensor " + name; return nullptr; }
    if (t->type != type || t->elems() != elems) {
        err = "tensor " + name + " has an unexpected type or size";
        return nullptr;
    }
    return t;
}

}  // namespace basalt::vision
