// A verified file stays read-only and cannot be replaced while the lease is held.
#pragma once

#include <memory>
#include <string>

namespace basalt::platform {

class VerifiedReadFile {
public:
    ~VerifiedReadFile();
    VerifiedReadFile(const VerifiedReadFile&) = delete;
    VerifiedReadFile& operator=(const VerifiedReadFile&) = delete;
    const std::string& path() const { return path_; }

private:
    VerifiedReadFile(void* handle, const std::string& path) : handle_(handle), path_(path) {}
    void* handle_ = nullptr;
    std::string path_;
    friend std::unique_ptr<VerifiedReadFile> verify_file_sha256(const std::string&, const std::string&, std::string&);
};

bool verified_file_supported();
std::unique_ptr<VerifiedReadFile> verify_file_sha256(const std::string& path, const std::string& expected,
                                                   std::string& err);

}  // namespace basalt::platform
