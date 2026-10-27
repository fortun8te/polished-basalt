#include "basalt/program/json_scan.hpp"
#include "basalt/program/tokenizer.hpp"
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

int main() {
    using namespace basalt::program;
    int failed = 0;
    auto check = [&](bool ok, const char* name) { if (!ok) { std::cerr << name << " failed\n"; ++failed; } };
    for (const auto& s : {std::string("9223372036854775808"), std::string(100, '9')}) {
        json::Scan sc{s}; int64_t n; check(!sc.integer(n), "oversized integer");
    }
    { json::Scan sc{"9223372036854775807"}; int64_t n; check(sc.integer(n) && n == (std::numeric_limits<int64_t>::max)(), "integer max"); }
    Tokenizer t; std::string err;
    auto reject = [&](const char* vocab, const char* merges, const char* types) {
        const bool result = t.load(vocab, merges, types, err);
        check(!result && !t.loaded(), "reject and unload");
    };
    check(t.load("{\"a\":0,\"b\":1,\"ab\":2}", "a b", "[1,1,1]", err), "valid vocab");
    reject("{\"a\":0,\"b\":18446744073709551615}", "", "[]");
    reject("{\"a\":2147483647}", "", "[]");
    reject("{\"a\":0,\"b\":0}", "", "[]");
    reject("{\"a\":-1}", "", "[]");
    reject("{\"a\":0}", "", "[257]");
    reject("{\"a\":0}", "", "[1");
    reject("{\"a\":0}", "x y", "[]");
    check(t.load("{\"a\":0,\"b\":1,\"ab\":2}", "a b", "[1,1,1]", err), "reload after failure");
    return failed != 0;
}
