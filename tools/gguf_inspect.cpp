// tools/gguf_inspect.cpp - CLI GGUF v3 parser and inspector.
//
// Built on the existing Strata parser (strata::GgufModel: header, typed metadata KV,
// multi-shard tensor directory, mmap) — no re-parsing of the GGUF format here.
//
// Usage:
//   gguf-inspect [options] <file.gguf>
//
// Options:
//   --kv PREFIX       only metadata keys starting with PREFIX
//   --tensor SUBSTR   only tensors whose name contains SUBSTR
//   --no-kv           skip the metadata section
//   --no-tensors      skip the tensor table
//   --types           show only the type histogram and totals
//
// Build (project dev-shell; NINFER_STRATA_SRC is the pinned Strata source tree):
//   nix develop path:. -c g++ -std=c++20 -O2 -I"$NINFER_STRATA_SRC/include" \
//       tools/gguf_inspect.cpp -o /tmp/gguf-inspect

#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string meta_type_name(strata::MetaType type) {
    switch (type) {
    case strata::MetaType::U8: return "U8";
    case strata::MetaType::I8: return "I8";
    case strata::MetaType::U16: return "U16";
    case strata::MetaType::I16: return "I16";
    case strata::MetaType::U32: return "U32";
    case strata::MetaType::I32: return "I32";
    case strata::MetaType::F32: return "F32";
    case strata::MetaType::BOOL: return "BOOL";
    case strata::MetaType::STRING: return "STRING";
    case strata::MetaType::ARRAY: return "ARRAY";
    case strata::MetaType::U64: return "U64";
    case strata::MetaType::I64: return "I64";
    case strata::MetaType::F64: return "F64";
    }
    return "?";
}

std::string meta_value_name(const strata::MetaValue& value) {
    switch (value.type) {
    case strata::MetaType::U8:
    case strata::MetaType::U16:
    case strata::MetaType::U32:
    case strata::MetaType::U64:
        return std::to_string(value.u);
    case strata::MetaType::I8:
    case strata::MetaType::I16:
    case strata::MetaType::I32:
    case strata::MetaType::I64:
        return std::to_string(static_cast<int64_t>(value.u));
    case strata::MetaType::F32:
    case strata::MetaType::F64:
        return std::format("{:g}", value.f);
    case strata::MetaType::BOOL:
        return value.u ? "true" : "false";
    case strata::MetaType::STRING:
        return "\"" + value.s + "\"";
    case strata::MetaType::ARRAY: {
        std::string out = "[" + meta_type_name(value.elem) + " x " + std::to_string(value.count) + "]";
        const auto limit = std::min<std::size_t>(value.items.size(), 8);
        for (std::size_t i = 0; i < limit; ++i) {
            out += (i ? " " : " ");
            out += meta_value_name(value.items[i]);
        }
        if (value.items.size() > limit) { out += " ..."; }
        return out;
    }
    }
    return "?";
}

} // namespace

int main(int argc, char** argv) {
    std::string path;
    std::string kv_prefix, tensor_sub;
    bool show_kv = true, show_tensors = true, types_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", arg.c_str()); exit(2); }
            return argv[++i];
        };
        if (arg == "--kv") { kv_prefix = next(); }
        else if (arg == "--tensor") { tensor_sub = next(); }
        else if (arg == "--no-kv") { show_kv = false; }
        else if (arg == "--no-tensors") { show_tensors = false; }
        else if (arg == "--types") { types_only = true; }
        else if (arg == "--help" || arg == "-h") {
            std::fprintf(stderr, "usage: gguf-inspect [--kv P] [--tensor S] [--no-kv] [--no-tensors] [--types] <file.gguf>\n");
            return 0;
        } else if (!arg.starts_with("-")) { path = arg; }
        else { std::fprintf(stderr, "unknown option %s\n", arg.c_str()); return 2; }
    }
    if (path.empty()) {
        std::fprintf(stderr, "usage: gguf-inspect <file.gguf>\n");
        return 2;
    }

    strata::GgufModel model = [&] {
        try {
            return strata::GgufModel::open(path);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "error: %s\n", error.what());
            std::exit(1);
        }
    }();
    std::printf("file: %s\n", path.c_str());
    std::printf("shards: %zu\n", model.size());
    for (std::size_t s = 0; s < model.size(); ++s) {
        const auto& shard = model.shard(s);
        std::printf("  shard %zu: %s (v%u, %zu tensors, data at %" PRIu64 " of %" PRIu64 ")\n",
                    s, shard.path().c_str(), shard.version(), shard.tensors().size(),
                    shard.data_start(), shard.file_size());
    }

    if (show_kv && !types_only) {
        const auto& meta = model.meta().metadata();
        std::printf("\nmetadata (%zu keys):\n", meta.size());
        for (const auto& [key, value] : meta) {
            if (!kv_prefix.empty() && key.rfind(kv_prefix, 0) != 0) { continue; }
            std::printf("  %-46s %-10s %s\n", key.c_str(), meta_type_name(value.type).c_str(),
                        meta_value_name(value).c_str());
        }
    }

    std::vector<std::pair<std::string, std::uint64_t>> histogram;
    auto bump = [&](uint32_t id, const char* name) {
        const std::string key = std::to_string(id) + " " + name;
        for (auto& entry : histogram) {
            if (entry.first == key) { ++entry.second; return; }
        }
        histogram.emplace_back(std::move(key), 1);
    };

    std::uint64_t total_tensors = 0, total_payload = 0, unknown = 0;
    if (show_tensors && !types_only) {
        std::printf("\ntensors:\n");
        std::printf("  %-6s %-52s %-10s %-6s %-28s %-14s %s\n",
                    "shard", "name", "type", "ndim", "shape (fastest first)", "payload", "offset");
        for (std::size_t s = 0; s < model.size(); ++s) {
            for (const auto& tensor : model.shard(s).tensors()) {
                if (!tensor_sub.empty() && tensor.name.find(tensor_sub) == std::string::npos) { continue; }
                const auto bytes = strata::tensor_payload_bytes(tensor);
                if (!bytes) { ++unknown; }
                total_tensors++;
                total_payload += bytes;
                bump(tensor.type, tensor.type_name());
                std::string shape;
                for (auto dim : tensor.shape) { shape += std::to_string(dim) + "x"; }
                if (!shape.empty()) { shape.pop_back(); }
                std::printf("  %-6zu %-52.52s %-10s %-6zu %-28s %-14" PRIu64 " %" PRIu64 "\n",
                            s, tensor.name.c_str(),
                            (std::to_string(tensor.type) + " " + tensor.type_name()).c_str(),
                            tensor.shape.size(), shape.c_str(), bytes, tensor.offset);
            }
        }
    } else {
        for (std::size_t s = 0; s < model.size(); ++s) {
            for (const auto& tensor : model.shard(s).tensors()) {
                total_tensors++;
                total_payload += strata::tensor_payload_bytes(tensor);
                bump(tensor.type, tensor.type_name());
            }
        }
    }

    std::sort(histogram.begin(), histogram.end());
    std::printf("\ntype histogram (%zu tensors, %" PRIu64 " payload bytes", total_tensors, total_payload);
    if (unknown) { std::printf(", %" PRIu64 " unknown-type tensors", unknown); }
    std::printf("):\n");
    for (const auto& [name, count] : histogram) {
        std::printf("  %-12s %" PRIu64 "\n", name.c_str(), count);
    }
    return 0;
}
