#pragma once

#include "artifact/framing.h"
#include "artifact/schema.h"
#include "core/weight_view.h"

#include <filesystem>
#include <memory>
#include <span>

namespace ninfer::artifact {

struct ReadSegment {
    std::size_t file_index           = 0;
    std::uint64_t file_offset        = 0;
    std::uint64_t destination_offset = 0;
    std::uint64_t bytes              = 0;
};

// Owns the cold directory and lazily opened files. Runtime views borrow materialized storage,
// never this Reader. Encodings are interpreted only for requested objects.
//
// Polymorphic base: the public path constructor reads a .ninfer v3 entry; derived readers
// (GgufReader) use the protected default constructor and override every directory and read
// method. The .ninfer behavior of the base is unchanged.
class Reader {
public:
    explicit Reader(const std::filesystem::path& path);
    virtual ~Reader();
    Reader(Reader&&) noexcept;
    Reader& operator=(Reader&&) noexcept;
    Reader(const Reader&)            = delete;
    Reader& operator=(const Reader&) = delete;

    [[nodiscard]] virtual const Directory& directory() const noexcept;
    [[nodiscard]] virtual const ArtifactId& artifact_id() const noexcept;
    [[nodiscard]] virtual std::uint64_t file_bytes() const noexcept;
    [[nodiscard]] virtual ObjectHandle find(std::string_view id) const;
    [[nodiscard]] virtual const WeightGeometry& geometry(ObjectHandle handle) const;
    virtual void validate_object(ObjectHandle handle) const;

    [[nodiscard]] virtual std::vector<ReadSegment> segments(std::uint64_t offset,
                                                            std::uint64_t bytes) const;
    virtual void read_into(std::uint64_t offset, std::span<std::byte> destination) const;
    [[nodiscard]] virtual std::vector<std::byte> read_range(std::uint64_t offset,
                                                            std::uint64_t bytes) const;
    [[nodiscard]] virtual std::vector<std::byte> read_object(ObjectHandle handle) const;
    [[nodiscard]] virtual std::size_t read_direct(std::size_t file_index, std::uint64_t file_offset,
                                                  std::span<std::byte> destination) const;

protected:
    // Base for derived non-.ninfer readers (defined in reader.cpp where Impl is
    // complete); only the public path constructor builds a .ninfer Impl eagerly.
    Reader();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::artifact
