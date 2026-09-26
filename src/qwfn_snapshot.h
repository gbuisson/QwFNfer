#pragma once

// Versioned, integrity-checked archive codec for durable QwFNfer snapshots.
// This layer is model-independent and uses only the C++20 standard library.

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace qwfn::snapshot {

inline constexpr std::array<uint8_t, 8> MAGIC = {'Q', 'W', 'F', 'N', 'S', 'N', 'P', 0};
inline constexpr uint32_t FORMAT_VERSION = 1;
inline constexpr size_t STREAM_CHUNK_BYTES = 4u << 20;

class sha256_hasher {
public:
    sha256_hasher();

    void update(const void * data, size_t size);
    std::array<uint8_t, 32> finish();

private:
    void transform(const uint8_t block[64]);

    std::array<uint32_t, 8> state_{};
    std::array<uint8_t, 64> block_{};
    uint64_t total_bytes_ = 0;
    size_t block_size_ = 0;
    bool finished_ = false;
};

std::array<uint8_t, 32> sha256(const void * data, size_t size);
std::string sha256_hex(const void * data, size_t size);

using read_callback = std::function<bool(uint64_t offset, void * dst, size_t size, std::string & err)>;
using write_callback = std::function<bool(uint64_t offset, const void * src, size_t size, std::string & err)>;

struct section_source {
    uint32_t kind = 0;
    uint32_t layer = 0;
    uint64_t length = 0;
    read_callback read;
};

struct section_info {
    uint32_t kind = 0;
    uint32_t layer = 0;
    uint64_t length = 0;
    uint64_t payload_offset = 0;
    std::array<uint8_t, 32> digest{};
};

struct archive_index {
    uint32_t format_version = 0;
    std::vector<uint8_t> compatibility;
    uint64_t body_length = 0;
    uint64_t file_size = 0;
    std::array<uint8_t, 32> body_digest{};
    std::vector<section_info> sections;
};

// Write to a mode-0600 sibling temporary, fsync it, atomically rename it onto
// path, then fsync the parent directory. Sources are streamed in bounded chunks.
bool write_atomic(
    const std::string & path,
    const std::vector<uint8_t> & compatibility,
    const std::vector<section_source> & sections,
    std::string & err);

// Parse and fully validate an archive without applying any state. This verifies
// magic/version/lengths/uniqueness, every section digest, the body digest, EOF,
// and rejects trailing bytes before returning an index.
bool inspect_and_validate(
    const std::string & path,
    archive_index & index,
    std::string & err);

// Stream a previously validated archive through caller-provided sinks. factory
// must reject unexpected sections or set sink empty to skip a known section.
// The archive identity and all hashes are checked again while applying. If this
// fails after a sink accepted bytes, the caller must reset its destination.
using sink_factory = std::function<bool(
    const section_info & section,
    write_callback & sink,
    std::string & err)>;

bool apply(
    const std::string & path,
    const archive_index & expected,
    const sink_factory & factory,
    std::string & err);

} // namespace qwfn::snapshot
