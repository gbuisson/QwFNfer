#include "qwfn_snapshot.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <limits>
#include <set>
#include <sstream>
#include <system_error>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace qwfn::snapshot {
namespace {

constexpr size_t FIXED_HEADER_BYTES = 64;
constexpr size_t SECTION_HEADER_BYTES = 48;
constexpr uint32_t MAX_SECTION_COUNT = 1u << 20;
constexpr uint32_t MAX_COMPATIBILITY_BYTES = 64u << 20;

uint32_t rotr(uint32_t value, unsigned count) {
    return (value >> count) | (value << (32 - count));
}

void put_u32(uint8_t * out, uint32_t value) {
    for (int i = 0; i < 4; ++i) out[i] = static_cast<uint8_t>(value >> (i * 8));
}

void put_u64(uint8_t * out, uint64_t value) {
    for (int i = 0; i < 8; ++i) out[i] = static_cast<uint8_t>(value >> (i * 8));
}

uint32_t get_u32(const uint8_t * in) {
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) value |= static_cast<uint32_t>(in[i]) << (i * 8);
    return value;
}

uint64_t get_u64(const uint8_t * in) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= static_cast<uint64_t>(in[i]) << (i * 8);
    return value;
}

std::string os_error(const char * operation) {
    return std::string(operation) + ": " + std::strerror(errno);
}

bool add_fits(uint64_t a, uint64_t b, uint64_t limit) {
    return a <= limit && b <= limit - a;
}

bool write_all(int fd, const void * data, size_t size, std::string & err) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    size_t done = 0;
    while (done < size) {
        const ssize_t n = ::write(fd, bytes + done, size - done);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            err = os_error("write snapshot");
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

bool pwrite_all(int fd, const void * data, size_t size, uint64_t offset, std::string & err) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    size_t done = 0;
    while (done < size) {
        const ssize_t n = ::pwrite(fd, bytes + done, size - done,
                                   static_cast<off_t>(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            err = os_error("rewrite snapshot header");
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

bool pread_all(int fd, void * data, size_t size, uint64_t offset, std::string & err) {
    auto * bytes = static_cast<uint8_t *>(data);
    size_t done = 0;
    while (done < size) {
        const ssize_t n = ::pread(fd, bytes + done, size - done,
                                  static_cast<off_t>(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) {
            err = os_error("read snapshot");
            return false;
        }
        if (n == 0) {
            err = "snapshot is truncated";
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

bool fd_size(int fd, uint64_t & size, std::string & err) {
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        err = os_error("stat snapshot");
        return false;
    }
    if (!S_ISREG(st.st_mode) || st.st_size < 0) {
        err = "snapshot is not a regular file";
        return false;
    }
    size = static_cast<uint64_t>(st.st_size);
    return true;
}

std::array<uint8_t, FIXED_HEADER_BYTES> encode_fixed_header(
        uint32_t compatibility_size, uint32_t section_count, uint64_t body_length,
        const std::array<uint8_t, 32> & digest) {
    std::array<uint8_t, FIXED_HEADER_BYTES> out{};
    std::copy(MAGIC.begin(), MAGIC.end(), out.begin());
    put_u32(out.data() + 8, FORMAT_VERSION);
    put_u32(out.data() + 12, compatibility_size);
    put_u32(out.data() + 16, section_count);
    put_u32(out.data() + 20, 0);
    put_u64(out.data() + 24, body_length);
    std::copy(digest.begin(), digest.end(), out.begin() + 32);
    return out;
}

std::array<uint8_t, SECTION_HEADER_BYTES> encode_section_header(
        uint32_t kind, uint32_t layer, uint64_t length,
        const std::array<uint8_t, 32> & digest) {
    std::array<uint8_t, SECTION_HEADER_BYTES> out{};
    put_u32(out.data(), kind);
    put_u32(out.data() + 4, layer);
    put_u64(out.data() + 8, length);
    std::copy(digest.begin(), digest.end(), out.begin() + 16);
    return out;
}

bool same_sections(const std::vector<section_info> & a, const std::vector<section_info> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].kind != b[i].kind || a[i].layer != b[i].layer ||
            a[i].length != b[i].length || a[i].payload_offset != b[i].payload_offset ||
            a[i].digest != b[i].digest) return false;
    }
    return true;
}

struct opened_archive {
    int fd = -1;
    uint64_t file_size = 0;
    uint32_t compatibility_size = 0;
    uint32_t section_count = 0;
    uint64_t body_length = 0;
    std::array<uint8_t, 32> body_digest{};
};

bool open_archive(const std::string & path, opened_archive & out, std::string & err) {
    out.fd = ::open(path.c_str(), O_RDONLY);
    if (out.fd < 0) {
        err = os_error("open snapshot");
        return false;
    }
    if (!fd_size(out.fd, out.file_size, err)) return false;
    if (out.file_size < FIXED_HEADER_BYTES) {
        err = "snapshot is shorter than its fixed header";
        return false;
    }
    std::array<uint8_t, FIXED_HEADER_BYTES> header{};
    if (!pread_all(out.fd, header.data(), header.size(), 0, err)) return false;
    if (!std::equal(MAGIC.begin(), MAGIC.end(), header.begin())) {
        err = "snapshot magic mismatch";
        return false;
    }
    const uint32_t version = get_u32(header.data() + 8);
    if (version != FORMAT_VERSION) {
        err = "unsupported snapshot format version " + std::to_string(version);
        return false;
    }
    out.compatibility_size = get_u32(header.data() + 12);
    out.section_count = get_u32(header.data() + 16);
    const uint32_t reserved = get_u32(header.data() + 20);
    out.body_length = get_u64(header.data() + 24);
    std::copy(header.begin() + 32, header.end(), out.body_digest.begin());
    if (reserved != 0) {
        err = "snapshot reserved header bits are non-zero";
        return false;
    }
    if (out.compatibility_size > MAX_COMPATIBILITY_BYTES) {
        err = "snapshot compatibility blob is too large";
        return false;
    }
    if (out.section_count > MAX_SECTION_COUNT) {
        err = "snapshot declares too many sections";
        return false;
    }
    if (out.body_length != out.file_size - FIXED_HEADER_BYTES) {
        err = out.body_length < out.file_size - FIXED_HEADER_BYTES
            ? "snapshot has trailing bytes" : "snapshot body length exceeds file size";
        return false;
    }
    const uint64_t minimum = static_cast<uint64_t>(out.compatibility_size) +
                             static_cast<uint64_t>(out.section_count) * SECTION_HEADER_BYTES;
    if (minimum > out.body_length) {
        err = "snapshot section table is truncated";
        return false;
    }
    return true;
}

void close_archive(opened_archive & archive) {
    if (archive.fd >= 0) ::close(archive.fd);
    archive.fd = -1;
}

} // namespace

sha256_hasher::sha256_hasher() {
    state_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
              0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
}

void sha256_hasher::transform(const uint8_t block[64]) {
    static constexpr uint32_t K[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u,
    };
    uint32_t w[64]{};
    for (int i = 0; i < 16; ++i) {
        w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
               (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
               (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
               static_cast<uint32_t>(block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = h + s1 + ch + K[i] + w[i];
        const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = s0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
    state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
}

void sha256_hasher::update(const void * data, size_t size) {
    if (finished_ || size == 0) return;
    const auto * bytes = static_cast<const uint8_t *>(data);
    total_bytes_ += size;
    while (size > 0) {
        const size_t take = std::min(size, block_.size() - block_size_);
        std::memcpy(block_.data() + block_size_, bytes, take);
        block_size_ += take;
        bytes += take;
        size -= take;
        if (block_size_ == block_.size()) {
            transform(block_.data());
            block_size_ = 0;
        }
    }
}

std::array<uint8_t, 32> sha256_hasher::finish() {
    if (!finished_) {
        const uint64_t bits = total_bytes_ * 8;
        block_[block_size_++] = 0x80;
        if (block_size_ > 56) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.end(), 0);
            transform(block_.data());
            block_size_ = 0;
        }
        std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.begin() + 56, 0);
        for (int i = 0; i < 8; ++i) block_[56 + i] = static_cast<uint8_t>(bits >> ((7 - i) * 8));
        transform(block_.data());
        finished_ = true;
    }
    std::array<uint8_t, 32> out{};
    for (size_t i = 0; i < state_.size(); ++i) {
        out[i * 4] = static_cast<uint8_t>(state_[i] >> 24);
        out[i * 4 + 1] = static_cast<uint8_t>(state_[i] >> 16);
        out[i * 4 + 2] = static_cast<uint8_t>(state_[i] >> 8);
        out[i * 4 + 3] = static_cast<uint8_t>(state_[i]);
    }
    return out;
}

std::array<uint8_t, 32> sha256(const void * data, size_t size) {
    sha256_hasher hasher;
    hasher.update(data, size);
    return hasher.finish();
}

std::string sha256_hex(const void * data, size_t size) {
    static constexpr char HEX[] = "0123456789abcdef";
    const auto digest = sha256(data, size);
    std::string out(64, '0');
    for (size_t i = 0; i < digest.size(); ++i) {
        out[i * 2] = HEX[digest[i] >> 4];
        out[i * 2 + 1] = HEX[digest[i] & 0x0f];
    }
    return out;
}

bool write_atomic(const std::string & path, const std::vector<uint8_t> & compatibility,
                  const std::vector<section_source> & sections, std::string & err) {
    err.clear();
    if (path.empty()) { err = "snapshot path is empty"; return false; }
    if (compatibility.size() > MAX_COMPATIBILITY_BYTES) { err = "compatibility blob is too large"; return false; }
    if (sections.size() > MAX_SECTION_COUNT) { err = "too many snapshot sections"; return false; }
    std::set<std::pair<uint32_t, uint32_t>> identities;
    for (const auto & section : sections) {
        if (!section.read && section.length != 0) { err = "snapshot section has no source callback"; return false; }
        if (!identities.emplace(section.kind, section.layer).second) { err = "duplicate snapshot section identity"; return false; }
    }

    const std::filesystem::path target(path);
    const std::filesystem::path parent = target.has_parent_path() ? target.parent_path() : std::filesystem::path(".");
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    if (ec) { err = "create snapshot directory: " + ec.message(); return false; }

    std::string temporary;
    int fd = -1;
    for (unsigned attempt = 0; attempt < 100 && fd < 0; ++attempt) {
        temporary = path + ".tmp." + std::to_string(static_cast<long long>(::getpid())) + "." + std::to_string(attempt);
        fd = ::open(temporary.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
        if (fd < 0 && errno != EEXIST) { err = os_error("create snapshot temporary"); return false; }
    }
    if (fd < 0) { err = "could not allocate a unique snapshot temporary"; return false; }
    bool renamed = false;
    auto cleanup = [&]() {
        if (fd >= 0) ::close(fd);
        fd = -1;
        if (!renamed) ::unlink(temporary.c_str());
    };

    std::array<uint8_t, 32> zero_digest{};
    const auto placeholder = encode_fixed_header(static_cast<uint32_t>(compatibility.size()),
                                                  static_cast<uint32_t>(sections.size()), 0, zero_digest);
    if (!write_all(fd, placeholder.data(), placeholder.size(), err) ||
        (!compatibility.empty() && !write_all(fd, compatibility.data(), compatibility.size(), err))) {
        cleanup(); return false;
    }

    std::vector<uint8_t> chunk(STREAM_CHUNK_BYTES);
    for (const auto & source : sections) {
        const off_t header_offset = ::lseek(fd, 0, SEEK_CUR);
        if (header_offset < 0) { err = os_error("seek snapshot"); cleanup(); return false; }
        const auto section_placeholder = encode_section_header(source.kind, source.layer, source.length, zero_digest);
        if (!write_all(fd, section_placeholder.data(), section_placeholder.size(), err)) { cleanup(); return false; }
        sha256_hasher section_hash;
        uint64_t offset = 0;
        while (offset < source.length) {
            const size_t amount = static_cast<size_t>(std::min<uint64_t>(chunk.size(), source.length - offset));
            if (!source.read(offset, chunk.data(), amount, err)) {
                if (err.empty()) err = "snapshot source callback failed";
                cleanup(); return false;
            }
            section_hash.update(chunk.data(), amount);
            if (!write_all(fd, chunk.data(), amount, err)) { cleanup(); return false; }
            offset += amount;
        }
        const auto section_digest = section_hash.finish();
        const auto encoded = encode_section_header(source.kind, source.layer, source.length, section_digest);
        if (!pwrite_all(fd, encoded.data(), encoded.size(), static_cast<uint64_t>(header_offset), err)) { cleanup(); return false; }
    }

    const off_t end_offset = ::lseek(fd, 0, SEEK_END);
    if (end_offset < 0) { err = os_error("seek snapshot end"); cleanup(); return false; }
    const uint64_t file_length = static_cast<uint64_t>(end_offset);
    if (file_length < FIXED_HEADER_BYTES) { err = "internal snapshot length underflow"; cleanup(); return false; }
    const uint64_t body_length = file_length - FIXED_HEADER_BYTES;
    sha256_hasher body_hash;
    uint64_t offset = FIXED_HEADER_BYTES;
    while (offset < file_length) {
        const size_t amount = static_cast<size_t>(std::min<uint64_t>(chunk.size(), file_length - offset));
        if (!pread_all(fd, chunk.data(), amount, offset, err)) { cleanup(); return false; }
        body_hash.update(chunk.data(), amount);
        offset += amount;
    }
    const auto final_header = encode_fixed_header(static_cast<uint32_t>(compatibility.size()),
                                                   static_cast<uint32_t>(sections.size()), body_length,
                                                   body_hash.finish());
    if (!pwrite_all(fd, final_header.data(), final_header.size(), 0, err)) { cleanup(); return false; }
    if (::fchmod(fd, 0600) != 0) { err = os_error("chmod snapshot"); cleanup(); return false; }
    if (::fsync(fd) != 0) { err = os_error("fsync snapshot"); cleanup(); return false; }
    if (::close(fd) != 0) { fd = -1; err = os_error("close snapshot"); cleanup(); return false; }
    fd = -1;
    if (::rename(temporary.c_str(), path.c_str()) != 0) { err = os_error("rename snapshot"); cleanup(); return false; }
    renamed = true;

    int directory_fd = ::open(parent.c_str(), O_RDONLY
#ifdef O_DIRECTORY
        | O_DIRECTORY
#endif
    );
    if (directory_fd < 0) { err = os_error("open snapshot directory"); cleanup(); return false; }
    const int sync_result = ::fsync(directory_fd);
    const int sync_errno = errno;
    ::close(directory_fd);
    if (sync_result != 0) { errno = sync_errno; err = os_error("fsync snapshot directory"); cleanup(); return false; }
    return true;
}

bool inspect_and_validate(const std::string & path, archive_index & index, std::string & err) {
    err.clear();
    index = {};
    opened_archive archive;
    if (!open_archive(path, archive, err)) { close_archive(archive); return false; }

    index.format_version = FORMAT_VERSION;
    index.body_length = archive.body_length;
    index.file_size = archive.file_size;
    index.body_digest = archive.body_digest;
    index.compatibility.resize(archive.compatibility_size);
    uint64_t cursor = FIXED_HEADER_BYTES;
    sha256_hasher body_hash;
    if (!index.compatibility.empty()) {
        if (!pread_all(archive.fd, index.compatibility.data(), index.compatibility.size(), cursor, err)) { close_archive(archive); return false; }
        body_hash.update(index.compatibility.data(), index.compatibility.size());
        cursor += index.compatibility.size();
    }

    std::set<std::pair<uint32_t, uint32_t>> identities;
    std::vector<uint8_t> chunk(STREAM_CHUNK_BYTES);
    index.sections.reserve(archive.section_count);
    for (uint32_t i = 0; i < archive.section_count; ++i) {
        if (!add_fits(cursor, SECTION_HEADER_BYTES, archive.file_size)) { err = "snapshot section header is truncated"; close_archive(archive); return false; }
        std::array<uint8_t, SECTION_HEADER_BYTES> encoded{};
        if (!pread_all(archive.fd, encoded.data(), encoded.size(), cursor, err)) { close_archive(archive); return false; }
        body_hash.update(encoded.data(), encoded.size());
        section_info info;
        info.kind = get_u32(encoded.data());
        info.layer = get_u32(encoded.data() + 4);
        info.length = get_u64(encoded.data() + 8);
        std::copy(encoded.begin() + 16, encoded.end(), info.digest.begin());
        cursor += SECTION_HEADER_BYTES;
        info.payload_offset = cursor;
        if (!identities.emplace(info.kind, info.layer).second) { err = "snapshot contains duplicate section identities"; close_archive(archive); return false; }
        if (!add_fits(cursor, info.length, archive.file_size)) { err = "snapshot section length exceeds file size"; close_archive(archive); return false; }
        sha256_hasher section_hash;
        uint64_t consumed = 0;
        while (consumed < info.length) {
            const size_t amount = static_cast<size_t>(std::min<uint64_t>(chunk.size(), info.length - consumed));
            if (!pread_all(archive.fd, chunk.data(), amount, cursor + consumed, err)) { close_archive(archive); return false; }
            section_hash.update(chunk.data(), amount);
            body_hash.update(chunk.data(), amount);
            consumed += amount;
        }
        if (section_hash.finish() != info.digest) { err = "snapshot section checksum mismatch"; close_archive(archive); return false; }
        cursor += info.length;
        index.sections.push_back(info);
    }
    if (cursor != archive.file_size) { err = cursor < archive.file_size ? "snapshot has trailing bytes" : "snapshot is truncated"; close_archive(archive); return false; }
    if (body_hash.finish() != archive.body_digest) { err = "snapshot body checksum mismatch"; close_archive(archive); return false; }
    close_archive(archive);
    return true;
}

bool apply(const std::string & path, const archive_index & expected,
           const sink_factory & factory, std::string & err) {
    err.clear();
    opened_archive archive;
    if (!open_archive(path, archive, err)) { close_archive(archive); return false; }
    if (expected.format_version != FORMAT_VERSION || expected.file_size != archive.file_size ||
        expected.body_length != archive.body_length || expected.body_digest != archive.body_digest ||
        expected.compatibility.size() != archive.compatibility_size ||
        expected.sections.size() != archive.section_count) {
        err = "snapshot changed after validation";
        close_archive(archive);
        return false;
    }

    uint64_t cursor = FIXED_HEADER_BYTES;
    std::vector<uint8_t> compatibility(archive.compatibility_size);
    sha256_hasher body_hash;
    if (!compatibility.empty()) {
        if (!pread_all(archive.fd, compatibility.data(), compatibility.size(), cursor, err)) { close_archive(archive); return false; }
        body_hash.update(compatibility.data(), compatibility.size());
        cursor += compatibility.size();
    }
    if (compatibility != expected.compatibility) { err = "snapshot compatibility metadata changed after validation"; close_archive(archive); return false; }

    std::vector<section_info> actual;
    actual.reserve(archive.section_count);
    std::set<std::pair<uint32_t, uint32_t>> identities;
    std::vector<uint8_t> chunk(STREAM_CHUNK_BYTES);
    for (uint32_t i = 0; i < archive.section_count; ++i) {
        std::array<uint8_t, SECTION_HEADER_BYTES> encoded{};
        if (!pread_all(archive.fd, encoded.data(), encoded.size(), cursor, err)) { close_archive(archive); return false; }
        body_hash.update(encoded.data(), encoded.size());
        section_info info;
        info.kind = get_u32(encoded.data());
        info.layer = get_u32(encoded.data() + 4);
        info.length = get_u64(encoded.data() + 8);
        std::copy(encoded.begin() + 16, encoded.end(), info.digest.begin());
        cursor += SECTION_HEADER_BYTES;
        info.payload_offset = cursor;
        if (!identities.emplace(info.kind, info.layer).second) { err = "snapshot contains duplicate section identities"; close_archive(archive); return false; }
        if (!add_fits(cursor, info.length, archive.file_size)) { err = "snapshot section length exceeds file size"; close_archive(archive); return false; }
        if (i >= expected.sections.size() ||
            info.kind != expected.sections[i].kind || info.layer != expected.sections[i].layer ||
            info.length != expected.sections[i].length || info.payload_offset != expected.sections[i].payload_offset ||
            info.digest != expected.sections[i].digest) {
            err = "snapshot section table changed after validation";
            close_archive(archive);
            return false;
        }
        write_callback sink;
        if (!factory(info, sink, err)) {
            if (err.empty()) err = "snapshot sink factory rejected a section";
            close_archive(archive);
            return false;
        }
        sha256_hasher section_hash;
        uint64_t consumed = 0;
        while (consumed < info.length) {
            const size_t amount = static_cast<size_t>(std::min<uint64_t>(chunk.size(), info.length - consumed));
            if (!pread_all(archive.fd, chunk.data(), amount, cursor + consumed, err)) { close_archive(archive); return false; }
            section_hash.update(chunk.data(), amount);
            body_hash.update(chunk.data(), amount);
            if (sink && !sink(consumed, chunk.data(), amount, err)) {
                if (err.empty()) err = "snapshot sink callback failed";
                close_archive(archive);
                return false;
            }
            consumed += amount;
        }
        if (section_hash.finish() != info.digest) { err = "snapshot section checksum mismatch while applying"; close_archive(archive); return false; }
        cursor += info.length;
        actual.push_back(info);
    }
    if (cursor != archive.file_size || !same_sections(actual, expected.sections)) {
        err = "snapshot structure changed after validation";
        close_archive(archive);
        return false;
    }
    if (body_hash.finish() != archive.body_digest) { err = "snapshot body checksum mismatch while applying"; close_archive(archive); return false; }
    close_archive(archive);
    return true;
}

} // namespace qwfn::snapshot
