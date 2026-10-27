#include "basalt/platform/verified_file.hpp"
#include "basalt/platform/file.hpp"

#include <vector>

#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#include <bcrypt.h>
#endif

namespace basalt::platform {

#if defined(_WIN32)
VerifiedReadFile::~VerifiedReadFile() {
    if (handle_ != nullptr) CloseHandle((HANDLE) handle_);
}

bool verified_file_supported() { return true; }

std::unique_ptr<VerifiedReadFile> verify_file_sha256(const std::string& path, const std::string& expected,
                                                   std::string& err) {
    const HANDLE file = CreateFileW(to_wide(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                    FILE_FLAG_SEQUENTIAL_SCAN | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (file == INVALID_HANDLE_VALUE) { err = "cannot open external file read-only without writers: " + path; return {}; }
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    auto fail = [&](const char* message) -> std::unique_ptr<VerifiedReadFile> {
        if (hash != nullptr) BCryptDestroyHash(hash);
        if (algorithm != nullptr) BCryptCloseAlgorithmProvider(algorithm, 0);
        CloseHandle(file);
        err = message;
        return {};
    };
    FILE_ATTRIBUTE_TAG_INFO attributes{};
    if (!GetFileInformationByHandleEx(file, FileAttributeTagInfo, &attributes, sizeof attributes) ||
        (attributes.FileAttributes & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY)) != 0)
        return fail("external hash source is not a regular file without reparse points");
    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
        return fail("cannot open the system SHA-256 provider");
    DWORD object_bytes = 0, returned = 0;
    if (BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH, (PUCHAR) &object_bytes, sizeof object_bytes, &returned, 0) < 0)
        return fail("cannot query SHA-256 object size");
    std::vector<unsigned char> object(object_bytes), buffer(4u << 20);
    if (BCryptCreateHash(algorithm, &hash, object.data(), object_bytes, nullptr, 0, 0) < 0)
        return fail("cannot create a SHA-256 hash");
    for (;;) {
        DWORD got = 0;
        if (!ReadFile(file, buffer.data(), (DWORD) buffer.size(), &got, nullptr))
            return fail("cannot read the external hash source");
        if (got == 0) break;
        if (BCryptHashData(hash, buffer.data(), got, 0) < 0) return fail("SHA-256 update failed");
    }
    unsigned char result[32]{};
    if (BCryptFinishHash(hash, result, sizeof result, 0) < 0) return fail("SHA-256 finish failed");
    std::string actual;
    const char hex[] = "0123456789abcdef";
    for (unsigned char byte : result) { actual += hex[byte >> 4]; actual += hex[byte & 15]; }
    if (actual != expected) return fail("external file SHA-256 differs from the required digest");
    std::vector<wchar_t> resolved(32768);
    const DWORD chars = GetFinalPathNameByHandleW(file, resolved.data(), (DWORD) resolved.size(), FILE_NAME_NORMALIZED);
    if (chars == 0 || chars >= resolved.size()) return fail("cannot resolve the verified file handle");
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, resolved.data(), (int) chars, nullptr, 0, nullptr, nullptr);
    if (bytes <= 0) return fail("cannot encode the verified file path");
    std::string verified_path((size_t) bytes, '\0');
    if (WideCharToMultiByte(CP_UTF8, 0, resolved.data(), (int) chars, verified_path.data(), bytes, nullptr, nullptr) != bytes)
        return fail("cannot encode the verified file path");
    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(algorithm, 0);
    return std::unique_ptr<VerifiedReadFile>(new VerifiedReadFile(file, verified_path));
}
#else
VerifiedReadFile::~VerifiedReadFile() = default;
bool verified_file_supported() { return false; }
std::unique_ptr<VerifiedReadFile> verify_file_sha256(const std::string&, const std::string&, std::string& err) {
    err = "compact packs with external PLE require the verified-file platform implementation";
    return {};
}
#endif

}  // namespace basalt::platform
