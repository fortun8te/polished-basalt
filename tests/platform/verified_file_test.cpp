#include "basalt/platform/verified_file.hpp"
#include "basalt/platform/file.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

int main() {
    if (!basalt::platform::verified_file_supported()) return 77;
    const fs::path root = fs::temp_directory_path() /
        ("basalt-verified-file-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(root);
    const std::string path = (root / "fixture.gguf").string();
    const std::string abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    int failures = 0;
    auto check = [&](bool ok, const char* label) { std::printf("%s %s\n", ok ? "ok" : "FAIL", label); if (!ok) ++failures; };
    { std::ofstream out(path, std::ios::binary); out << "abc"; }
    std::string err;
    auto lease = basalt::platform::verify_file_sha256(path, abc, err);
    check(lease != nullptr, "system SHA-256 accepts abc");
    if (lease != nullptr) {
        std::FILE* reader = basalt::platform::open_file_utf8(lease->path(), "rb");
        char bytes[3]{};
        const bool readable = reader != nullptr && std::fread(bytes, 1, sizeof bytes, reader) == sizeof bytes &&
                              std::string(bytes, sizeof bytes) == "abc";
        check(readable, "verified resolved path allows a read-only reader");
        if (reader != nullptr) std::fclose(reader);
        std::error_code operation;
        fs::rename(path, root / "renamed.gguf", operation);
        check(bool(operation), "verified lease prevents rename");
        operation.clear();
        const bool removed = fs::remove(path, operation);
        check(!removed && bool(operation), "verified lease prevents deletion");
    }
    std::FILE* writer = basalt::platform::open_file_utf8(path, "wb");
    check(writer == nullptr, "verified lease prevents another writer");
    if (writer != nullptr) std::fclose(writer);
    lease.reset();
    std::error_code operation;
    fs::rename(path, root / "released.gguf", operation);
    check(!operation, "released lease allows rename");
    if (!operation) fs::rename(root / "released.gguf", path, operation);
    check(!operation, "released fixture name restored");
    { std::ofstream out(path, std::ios::binary); out << "abd"; }
    check(basalt::platform::verify_file_sha256(path, abc, err) == nullptr, "one changed byte rejected");
    const std::string header = "GGUF IQ4_NL shape=160,320001536 ";
    { std::ofstream out(path, std::ios::binary); out << header << "AAA"; }
    // Digest of the same header and AAA payload, compared against a same-length BBB payload below.
    const std::string table_sha = "30b10cb8291a8ec216b1faaa642896e971209347a65e9430511a039601cefd15";
    check(basalt::platform::verify_file_sha256(path, table_sha, err) != nullptr, "table fixture accepted");
    { std::ofstream out(path, std::ios::binary); out << header << "BBB"; }
    check(basalt::platform::verify_file_sha256(path, table_sha, err) == nullptr, "same-geometry wrong table rejected");
    fs::remove_all(root, operation);
    check(!operation && !fs::exists(root), "released leases allow cleanup");
    return failures ? 1 : 0;
}
