// An external PLE is accepted only by a compact pack with an explicit source contract.
#pragma once

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/core/model_file.hpp"

#include <algorithm>
#include <string>

namespace basalt::core {

struct ExternalPleContract {
    bool active = false;
    uint64_t source_tensor_offset = 0;
    uint64_t source_file_bytes = 0;
    std::string source_file_sha256, tensor_sha256, reconstructed_gguf_sha256;
};

inline bool external_ple_uint(const Json* value, uint64_t expected) {
    return value != nullptr && value->type == Json::Type::Int && !value->neg && value->u == expected;
}

inline bool external_ple_hash(const Json* value) {
    return value != nullptr && value->is_string() && value->s.size() == 64 &&
           std::all_of(value->s.begin(), value->s.end(), [](char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
}

inline bool external_ple_contract(const Json& meta, const std::string& path, ExternalPleContract& out,
                                  std::string& err) {
    out = {};
    const Json* contract = meta.get("external_ple");
    if (contract == nullptr) {
        if (path.empty()) return true;
        err = "--ple-gguf requires a compact pack with an external_ple contract";
        return false;
    }
    if (path.empty()) {
        err = "this compact pack requires --ple-gguf FILE";
        return false;
    }
    const Json* tensors = meta.get("tensors");
    const Json* shape = contract->get("shape");
    const Json* type = contract->get("type");
    const Json* name = contract->get("tensor");
    if (!contract->is_object() || tensors == nullptr || !tensors->is_object() ||
        tensors->get("per_layer_token_embd.weight") != nullptr ||
        !external_ple_uint(contract->get("version"), 1) || name == nullptr || !name->is_string() ||
        name->s != "per_layer_token_embd.weight" || type == nullptr || !type->is_string() || type->s != "IQ4_NL" ||
        !external_ple_uint(contract->get("type_id"), 20) || shape == nullptr || !shape->is_array() ||
        shape->count != 2 || shape->items.size() != 2 || !external_ple_uint(&shape->items[0], 160) ||
        !external_ple_uint(&shape->items[1], 320001536) ||
        !external_ple_uint(contract->get("bytes"), 28800138240ull) ||
        !external_ple_uint(contract->get("source_tensor_offset"), 192) ||
        !external_ple_uint(contract->get("source_file_bytes"), 28800138432ull) ||
        !external_ple_hash(contract->get("source_file_sha256")) ||
        contract->get("source_file_sha256")->s != "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113" ||
        !external_ple_hash(contract->get("tensor_sha256")) ||
        !external_ple_hash(contract->get("reconstructed_gguf_sha256"))) {
        err = "invalid external_ple contract or duplicate embedded PLE tensor";
        return false;
    }
    out.active = true;
    out.source_tensor_offset = 192;
    out.source_file_bytes = 28800138432ull;
    out.source_file_sha256 = contract->get("source_file_sha256")->s;
    out.tensor_sha256 = contract->get("tensor_sha256")->s;
    out.reconstructed_gguf_sha256 = contract->get("reconstructed_gguf_sha256")->s;
    return true;
}

// Used after the source hash lease has been acquired, before the table reader opens it.
inline bool external_ple_tensor(const ExternalPleContract& contract, const TensorInfo* tensor,
                                uint64_t data_start, uint64_t file_bytes, std::string& err) {
    if (!contract.active || tensor == nullptr || tensor->name != "per_layer_token_embd.weight" ||
        tensor->type != 20 || tensor->shape != std::vector<uint64_t>{160, 320001536} ||
        data_start > contract.source_tensor_offset ||
        tensor->offset != contract.source_tensor_offset - data_start || file_bytes != contract.source_file_bytes ||
        contract.source_tensor_offset > file_bytes || 28800138240ull != file_bytes - contract.source_tensor_offset) {
        err = "--ple-gguf tensor type, dimensions, offset or file size differs from the compact pack contract";
        return false;
    }
    return true;
}

}  // namespace basalt::core
