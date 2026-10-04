#pragma once

#include "artifact/reader.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::artifact {

// ggml type id <-> QType for the GGUF block encodings this engine supports. The mapping
// is bidirectional and total over the supported ggml ids (table in gguf.cpp); the GGUF
// float types reuse the existing .ninfer QTypes (ggml F32 is QType::FP32, ggml BF16 is
// QType::BF16). Unsupported ggml ids and non-GGUF QTypes throw ArtifactError.
[[nodiscard]] QType gguf_qtype(std::uint32_t ggml_type_id);
[[nodiscard]] std::uint32_t ggml_type(QType format);

// NInfer logical names of one GGUF tensor (empty for names with no NInfer counterpart).
// `mtp_block` is the nextn (MTP) block index, qwen35.block_count - 1: its tensors take
// the "mtp/" prefix, every earlier block takes "text/layers/{i}/".
[[nodiscard]] std::vector<std::string> logical_names(const std::string& gguf_name,
                                                     std::size_t mtp_block);

// Reads a (possibly split) GGUF v3 model through strata::GgufModel and exposes it as a
// synthetic NInfer Directory: one TensorObject per GGUF tensor (dims reversed into NInfer
// order), NInfer logical-name aliases in object_index, one FileRecord per shard, and a
// components/metadata JSON derived from the GGUF KV. The logical payload is the
// concatenation of each shard's data section [data_start, file_size); the directory is
// built eagerly in the constructor, so every base-class read method works unchanged.
class GgufReader : public Reader {
public:
    explicit GgufReader(const std::filesystem::path& path);
    ~GgufReader() override;
    GgufReader(GgufReader&&) noexcept;
    GgufReader& operator=(GgufReader&&) noexcept;
    GgufReader(const GgufReader&)            = delete;
    GgufReader& operator=(const GgufReader&) = delete;

    [[nodiscard]] const Directory& directory() const noexcept override;
    [[nodiscard]] const ArtifactId& artifact_id() const noexcept override;
    [[nodiscard]] std::uint64_t file_bytes() const noexcept override;
    [[nodiscard]] ObjectHandle find(std::string_view id) const override;
    [[nodiscard]] const WeightGeometry& geometry(ObjectHandle handle) const override;
    void validate_object(ObjectHandle handle) const override;
    [[nodiscard]] std::vector<ReadSegment> segments(std::uint64_t offset,
                                                    std::uint64_t bytes) const override;
    void read_into(std::uint64_t offset, std::span<std::byte> destination) const override;
    [[nodiscard]] std::vector<std::byte> read_range(std::uint64_t offset,
                                                    std::uint64_t bytes) const override;
    [[nodiscard]] std::vector<std::byte> read_object(ObjectHandle handle) const override;
    [[nodiscard]] std::size_t read_direct(std::size_t file_index, std::uint64_t file_offset,
                                          std::span<std::byte> destination) const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::artifact
