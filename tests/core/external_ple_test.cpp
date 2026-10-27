#include "basalt/core/external_ple.hpp"

#include <cstdio>
#include <string>

using namespace basalt::core;

namespace {
int failures = 0;
void check(bool ok, const char* label) {
    std::printf("%s %s\n", ok ? "ok" : "FAIL", label);
    if (!ok) ++failures;
}
Json parsed(const std::string& text) {
    Json result;
    std::string err;
    check(parse_json(text.data(), text.size(), result, err), "parse fixture JSON");
    return result;
}
std::string fixture() {
    const std::string hash(64, 'a');
    return "{\"tensors\":{},\"external_ple\":{\"version\":1,\"tensor\":\"per_layer_token_embd.weight\","
           "\"type\":\"IQ4_NL\",\"type_id\":20,\"shape\":[160,320001536],\"bytes\":28800138240,"
           "\"source_tensor_offset\":192,\"source_file_bytes\":28800138432,\"source_file_sha256\":\""
           "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"
           "\",\"tensor_sha256\":\"" + hash + "\",\"reconstructed_gguf_sha256\":\"" + hash + "\"}}";
}
}  // namespace

int main() {
    ExternalPleContract contract;
    std::string err;
    check(external_ple_contract(parsed("{\"tensors\":{}}"), "", contract, err) && !contract.active,
          "ordinary pack keeps embedded source");
    check(!external_ple_contract(parsed("{\"tensors\":{}}"), "table.gguf", contract, err),
          "external override without marker refused");
    const Json good = parsed(fixture());
    check(!external_ple_contract(good, "", contract, err), "compact pack without external path refused");
    check(external_ple_contract(good, "table.gguf", contract, err) && contract.active, "valid contract accepted");
    check(contract.source_file_sha256 == "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113" &&
          contract.tensor_sha256 == std::string(64, 'a') && contract.reconstructed_gguf_sha256 == std::string(64, 'a'),
          "contract digests retained");
    basalt::TensorInfo tensor;
    tensor.name = "per_layer_token_embd.weight";
    tensor.type = 20;
    tensor.shape = {160, 320001536};
    check(external_ple_tensor(contract, &tensor, 192, 28800138432ull, err), "original tensor source accepted");
    check(!external_ple_tensor(contract, nullptr, 192, 28800138432ull, err), "missing tensor refused");
    check(!external_ple_tensor(contract, &tensor, 0, 28800138432ull, err), "wrong data offset refused");
    check(!external_ple_tensor(contract, &tensor, 192, 28800138431ull, err), "truncated source refused");
    tensor.type = 8;
    check(!external_ple_tensor(contract, &tensor, 192, 28800138432ull, err), "wrong quantization refused");
    tensor.type = 20;
    tensor.shape[1] = 1;
    check(!external_ple_tensor(contract, &tensor, 192, 28800138432ull, err), "wrong dimensions refused");
    for (const std::string field : {"version", "bytes", "source_tensor_offset"}) {
        Json bad = good;
        bad.members["external_ple"].members[field].u++;
        check(!external_ple_contract(bad, "table.gguf", contract, err), field.c_str());
    }
    Json duplicate = good;
    duplicate.members["tensors"].members["per_layer_token_embd.weight"] = Json{};
    check(!external_ple_contract(duplicate, "table.gguf", contract, err), "duplicate embedded table refused");
    Json malformed = good;
    malformed.members["external_ple"].members["tensor_sha256"].s = "not-a-hash";
    check(!external_ple_contract(malformed, "table.gguf", contract, err), "malformed hash refused");
    Json unpinned = good;
    unpinned.members["external_ple"].members["source_file_sha256"].s = std::string(64, 'a');
    check(!external_ple_contract(unpinned, "table.gguf", contract, err), "unpinned source hash refused");
    return failures ? 1 : 0;
}
