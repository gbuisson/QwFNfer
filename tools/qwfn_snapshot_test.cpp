#include "qwfn_snapshot.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace qwfn::snapshot;

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<uint8_t> read_bytes(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "open fixture for read");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void write_bytes(const fs::path & path, const std::vector<uint8_t> & bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(output.good(), "open fixture for write");
    output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    require(output.good(), "write fixture");
}

void put_u32(std::vector<uint8_t> & bytes, size_t offset, uint32_t value) {
    require(offset + 4 <= bytes.size(), "put_u32 range");
    for (int i = 0; i < 4; ++i) bytes[offset + i] = static_cast<uint8_t>(value >> (i * 8));
}

void refresh_body_digest(std::vector<uint8_t> & bytes) {
    require(bytes.size() >= 64, "archive fixed header");
    const auto digest = sha256(bytes.data() + 64, bytes.size() - 64);
    std::copy(digest.begin(), digest.end(), bytes.begin() + 32);
}

section_source memory_source(uint32_t kind, uint32_t layer, const std::vector<uint8_t> & data) {
    return section_source{
        kind,
        layer,
        data.size(),
        [&data](uint64_t offset, void * dst, size_t size, std::string & err) {
            if (offset > data.size() || size > data.size() - static_cast<size_t>(offset)) {
                err = "memory source range";
                return false;
            }
            std::memcpy(dst, data.data() + offset, size);
            return true;
        },
    };
}

void expect_invalid(const fs::path & path, const std::string & label) {
    archive_index index;
    std::string err;
    require(!inspect_and_validate(path.string(), index, err), label + " unexpectedly validated");
    require(!err.empty(), label + " returned no error");
}

void test_sha256() {
    require(sha256_hex("", 0) ==
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "SHA-256 empty vector");
    const std::string abc = "abc";
    require(sha256_hex(abc.data(), abc.size()) ==
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "SHA-256 abc vector");
}

void test_round_trip(const fs::path & root) {
    const fs::path path = root / "roundtrip.bin";
    const std::vector<uint8_t> compatibility = {'m','o','d','e','l',0,1,2};
    std::vector<uint8_t> first((4u << 20) + 137);
    std::vector<uint8_t> second(7777);
    for (size_t i = 0; i < first.size(); ++i) first[i] = static_cast<uint8_t>((i * 17 + 3) & 0xff);
    for (size_t i = 0; i < second.size(); ++i) second[i] = static_cast<uint8_t>((i * 29 + 11) & 0xff);
    const std::vector<uint8_t> empty;

    std::string err;
    require(write_atomic(path.string(), compatibility,
                         {memory_source(1, 3, first), memory_source(2, 7, second), memory_source(9, 0, empty)}, err),
            "write roundtrip: " + err);

    struct stat st {};
    require(::stat(path.c_str(), &st) == 0, "stat roundtrip");
    require((st.st_mode & 0777) == 0600, "snapshot mode is not 0600");

    archive_index index;
    require(inspect_and_validate(path.string(), index, err), "inspect roundtrip: " + err);
    require(index.compatibility == compatibility, "compatibility round trip");
    require(index.sections.size() == 3, "section count round trip");

    std::map<std::pair<uint32_t,uint32_t>, std::vector<uint8_t>> output;
    require(apply(path.string(), index,
        [&output](const section_info & info, write_callback & sink, std::string &) {
            auto & bytes = output[{info.kind, info.layer}];
            bytes.resize(static_cast<size_t>(info.length));
            sink = [&bytes](uint64_t offset, const void * src, size_t size, std::string & sink_err) {
                if (offset > bytes.size() || size > bytes.size() - static_cast<size_t>(offset)) {
                    sink_err = "memory sink range";
                    return false;
                }
                std::memcpy(bytes.data() + offset, src, size);
                return true;
            };
            return true;
        }, err), "apply roundtrip: " + err);
    require(output[std::make_pair(1u,3u)] == first, "first section round trip");
    require(output[std::make_pair(2u,7u)] == second, "second section round trip");
    require(output[std::make_pair(9u,0u)].empty(), "empty section round trip");

    std::vector<uint8_t> replacement = {9,8,7,6};
    require(write_atomic(path.string(), {'v','2'}, {memory_source(4, 4, replacement)}, err),
            "atomic replacement: " + err);
    archive_index replaced;
    require(inspect_and_validate(path.string(), replaced, err), "inspect replacement: " + err);
    require(replaced.compatibility == std::vector<uint8_t>({'v','2'}), "replacement compatibility");
    require(replaced.sections.size() == 1 && replaced.sections[0].kind == 4, "replacement sections");
    require(::stat(path.c_str(), &st) == 0 && (st.st_mode & 0777) == 0600, "replacement mode is not 0600");
}

void test_fail_closed(const fs::path & root) {
    const std::vector<uint8_t> one = {1,2,3,4,5};
    const std::vector<uint8_t> two = {6,7,8,9};
    const fs::path base = root / "base.bin";
    std::string err;
    require(write_atomic(base.string(), {'c'}, {memory_source(1, 1, one), memory_source(2, 2, two)}, err),
            "write fail-closed fixture: " + err);
    archive_index base_index;
    require(inspect_and_validate(base.string(), base_index, err), "inspect fail-closed fixture: " + err);

    auto copy = [&](const std::string & name) {
        const fs::path target = root / name;
        fs::copy_file(base, target, fs::copy_options::overwrite_existing);
        return target;
    };

    {
        const fs::path path = copy("bad-magic.bin");
        auto bytes = read_bytes(path); bytes[0] ^= 0xff; write_bytes(path, bytes);
        expect_invalid(path, "bad magic");
    }
    {
        const fs::path path = copy("bad-version.bin");
        auto bytes = read_bytes(path); put_u32(bytes, 8, FORMAT_VERSION + 1); write_bytes(path, bytes);
        expect_invalid(path, "bad version");
    }
    {
        const fs::path path = copy("corrupt.bin");
        auto bytes = read_bytes(path);
        bytes[static_cast<size_t>(base_index.sections[0].payload_offset)] ^= 0x40;
        write_bytes(path, bytes);
        expect_invalid(path, "corrupt payload");
    }
    {
        const fs::path path = copy("truncated.bin");
        fs::resize_file(path, fs::file_size(path) - 1);
        expect_invalid(path, "truncation");
    }
    {
        const fs::path path = copy("trailing.bin");
        std::ofstream output(path, std::ios::binary | std::ios::app); output.put('\0'); output.close();
        expect_invalid(path, "trailing bytes");
    }
    {
        const fs::path path = copy("duplicate.bin");
        auto bytes = read_bytes(path);
        const size_t second_header = static_cast<size_t>(base_index.sections[1].payload_offset - 48);
        put_u32(bytes, second_header, base_index.sections[0].kind);
        put_u32(bytes, second_header + 4, base_index.sections[0].layer);
        refresh_body_digest(bytes);
        write_bytes(path, bytes);
        expect_invalid(path, "duplicate section");
    }
    {
        const fs::path path = copy("changed-after-validation.bin");
        auto bytes = read_bytes(path);
        bytes[static_cast<size_t>(base_index.sections[0].payload_offset)] ^= 0x01;
        write_bytes(path, bytes);
        bool sink_called = false;
        require(!apply(path.string(), base_index,
            [&sink_called](const section_info &, write_callback & sink, std::string &) {
                sink = [&sink_called](uint64_t, const void *, size_t, std::string &) { sink_called = true; return true; };
                return true;
            }, err), "changed archive unexpectedly applied");
        require(!err.empty(), "changed archive apply returned no error");
        require(sink_called, "apply corruption test did not exercise partial-write contract");
    }
}

} // namespace

int main() {
    fs::path root = fs::temp_directory_path() /
        ("qwfn-snapshot-test-" + std::to_string(static_cast<long long>(::getpid())));
    try {
        fs::remove_all(root);
        fs::create_directories(root);
        test_sha256();
        test_round_trip(root);
        test_fail_closed(root);
        fs::remove_all(root);
        std::cout << "qwfn-snapshot-test: all tests passed\n";
        return 0;
    } catch (const std::exception & ex) {
        fs::remove_all(root);
        std::cerr << "qwfn-snapshot-test: " << ex.what() << '\n';
        return 1;
    }
}
