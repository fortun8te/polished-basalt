#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/core/model_file.hpp"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
using namespace basalt;
using namespace basalt::core;
template <typename T> void put(std::vector<uint8_t>& v, T x) { auto p = reinterpret_cast<const uint8_t*>(&x); v.insert(v.end(), p, p + sizeof x); }
void str(std::vector<uint8_t>& v, const std::string& s) { put<uint64_t>(v, s.size()); v.insert(v.end(), s.begin(), s.end()); }
int main() {
    int fail = 0;
    auto check = [&](bool ok, const char* name) { if (!ok) { std::cerr << name << " failed\n"; ++fail; } };
    const auto dir = std::filesystem::temp_directory_path() / ("basalt-hardening-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    auto gguf = [&](const std::vector<uint8_t>& v, bool ok) {
        auto p = dir / "test.gguf"; { std::ofstream f(p, std::ios::binary); f.write((const char*)v.data(), (std::streamsize)v.size()); }
        bool accepted = false; try { GgufFile g(p.string()); accepted = true; } catch (const std::exception&) {}
        check(accepted == ok, "gguf rejection");
    };
    auto header = [](uint64_t nt, uint64_t nk) { std::vector<uint8_t> v; put<uint32_t>(v, 0x46554747); put<uint32_t>(v, 3); put(v, nt); put(v, nk); return v; };
    auto v = header(0, 0); v.resize(32); gguf(v, true);
    gguf(header(UINT64_MAX, 0), false);
    gguf(header(0, UINT64_MAX), false);
    v = header(0, 1); put<uint64_t>(v, UINT64_MAX); put<uint32_t>(v, 0); gguf(v, false);
    v = header(0, 1); str(v, "general.alignment"); put<uint32_t>(v, 10); put<uint64_t>(v, UINT64_MAX); gguf(v, false);
    TensorInfo ti; ti.type = 0; ti.shape = {UINT64_MAX, 2}; check(ti.elements() == 0 && tensor_payload_bytes(ti) == 0, "element overflow");
    Json parsed; std::string err;
    std::string deep(66, '['); deep += '0'; deep.append(66, ']');
    check(!parse_json(deep.data(), deep.size(), parsed, err), "json depth");
    char byte = 0;
    check(!read_range_at(FileRange("unused", UINT64_MAX, 0), 1, &byte, 1, err), "range offset overflow");
    auto hc = dir / "bad.hc8";
    { Hc8Header h{}; std::memcpy(h.magic, "BASALTH8", 8); h.version = 1; h.n_entries = 1;
      Hc8Entry e{}; e.rows = UINT32_MAX; e.cols = UINT32_MAX - 31; e.srow = UINT32_MAX; e.offset = sizeof h + sizeof e;
      std::ofstream f(hc, std::ios::binary); f.write((char*)&h, sizeof h); f.write((char*)&e, sizeof e); }
    std::map<std::string, Hc8Copy> out; check(!hc8_read(FileRange(hc.string()), out, err), "hc8 overflow");
    auto view = [&](const std::string& tensor, bool expected) {
        const std::string meta = "{\"gguf_kv\":{\"negative\":-9223372036854775808},\"tensors\":{\"x\":" + tensor + "}}";
        const auto path = dir / "test.basalt";
        FileHeader h{}; std::memcpy(h.magic, kModelMagic, sizeof kModelMagic); h.version = kModelVersion;
        h.n_sections = 2; h.file_bytes = 2 * kSectionAlign + 32;
        SectionEntry m{}, g{}; std::memcpy(m.name, "meta", 4); m.offset = kSectionAlign; m.bytes = meta.size();
        std::memcpy(g.name, "gguf", 4); g.offset = 2 * kSectionAlign; g.bytes = 32;
        { std::ofstream f(path, std::ios::binary); f.write((char*)&h, sizeof h); f.write((char*)&m, sizeof m);
          f.write((char*)&g, sizeof g); f.seekp(m.offset); f.write(meta.data(), (std::streamsize)meta.size());
          f.seekp(h.file_bytes - 1); f.put(0); }
        ModelFile model;
        check(model.open(path.string(), err) && model.load_meta(err), "model fixture");
        check(model.register_gguf_view(err) == expected, "model tensor bounds");
    };
    view("{\"type_id\":0,\"shape\":[8],\"offset\":0,\"bytes\":32}", true);
    view("{\"type_id\":0,\"shape\":[8],\"offset\":9999999999999999999}", false);
    view("{\"type_id\":0,\"shape\":[8],\"offset\":-1}", false);
    view("{\"type_id\":0,\"shape\":[9999999999999999999,3],\"offset\":0}", false);
    view("{\"type_id\":0,\"shape\":[9],\"offset\":0}", false);
    std::filesystem::remove_all(dir);
    return fail != 0;
}
