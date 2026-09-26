#include "qwfn_engine.h"
#include "qwfn_snapshot.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <utility>

#include <sys/stat.h>

namespace qwfn {
namespace {

enum section_kind : uint32_t {
    SEC_TOKENS       = 1,
    SEC_SERVER_STATE = 2,
    SEC_K_CACHE      = 10,
    SEC_V_CACHE      = 11,
    SEC_INDEX_CACHE  = 12,
    SEC_RECURRENT    = 20,
    SEC_CONV_HISTORY = 21,
    SEC_PLE_HISTORY  = 30,
};

using key = std::pair<uint32_t, uint32_t>;

void append_u32(std::vector<uint8_t> & out, uint32_t value) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(value >> (i * 8)));
}
void append_i32(std::vector<uint8_t> & out, int32_t value) {
    append_u32(out, static_cast<uint32_t>(value));
}
void append_u64(std::vector<uint8_t> & out, uint64_t value) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(value >> (i * 8)));
}
void append_float(std::vector<uint8_t> & out, float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    append_u32(out, bits);
}
void append_string(std::vector<uint8_t> & out, const std::string & value) {
    append_u64(out, value.size());
    out.insert(out.end(), value.begin(), value.end());
}
template <typename T>
void append_vector32(std::vector<uint8_t> & out, const std::vector<T> & values) {
    append_u64(out, values.size());
    for (const auto value : values) append_u32(out, static_cast<uint32_t>(value));
}
void append_vector64(std::vector<uint8_t> & out, const std::vector<uint64_t> & values) {
    append_u64(out, values.size());
    for (const auto value : values) append_u64(out, value);
}

std::vector<uint8_t> encode_tokens(const std::vector<int32_t> & tokens) {
    std::vector<uint8_t> bytes;
    bytes.reserve(tokens.size() * 4);
    for (const int32_t token : tokens) append_i32(bytes, token);
    return bytes;
}

bool read_section(const std::string & path, const snapshot::section_info & section,
                  std::vector<uint8_t> & bytes, std::string & err) {
    if (section.length > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        err = "snapshot metadata section is too large";
        return false;
    }
    bytes.resize(static_cast<size_t>(section.length));
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        err = "open snapshot metadata: " + std::string(std::strerror(errno));
        return false;
    }
    input.seekg(static_cast<std::streamoff>(section.payload_offset));
    if (!input) {
        err = "seek snapshot metadata failed";
        return false;
    }
    if (!bytes.empty()) input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input) {
        err = "read snapshot metadata failed";
        return false;
    }
    return true;
}

snapshot::section_source byte_source(uint32_t kind, uint32_t layer,
                                     std::shared_ptr<const std::vector<uint8_t>> bytes) {
    return snapshot::section_source{
        kind, layer, bytes->size(),
        [bytes = std::move(bytes)](uint64_t offset, void * dst, size_t size, std::string & err) {
            if (offset > bytes->size() || size > bytes->size() - static_cast<size_t>(offset)) {
                err = "snapshot byte source range";
                return false;
            }
            std::memcpy(dst, bytes->data() + offset, size);
            return true;
        },
    };
}

snapshot::section_source tensor_source(uint32_t kind, uint32_t layer,
                                       ggml_tensor * tensor, uint64_t length) {
    return snapshot::section_source{
        kind, layer, length,
        [tensor](uint64_t offset, void * dst, size_t size, std::string & err) {
            if (offset > ggml_nbytes(tensor) || size > ggml_nbytes(tensor) - static_cast<size_t>(offset)) {
                err = "snapshot tensor source range";
                return false;
            }
            ggml_backend_tensor_get(tensor, dst, static_cast<size_t>(offset), size);
            return true;
        },
    };
}

uint64_t prefix_bytes(ggml_tensor * tensor, uint64_t row_elements, uint64_t rows) {
    if (!tensor) return 0;
    return static_cast<uint64_t>(ggml_row_size(tensor->type, row_elements)) * rows;
}

std::string digest_hex(const std::array<uint8_t, 32> & digest) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (const uint8_t byte : digest) {
        out.push_back(hex[byte >> 4]);
        out.push_back(hex[byte & 15]);
    }
    return out;
}

} // namespace

std::vector<uint8_t> engine::snapshot_compatibility() const {
    std::vector<uint8_t> out;
    append_string(out, "qwfnfer-engine-snapshot-v1");

    auto model_fingerprint = [](const model_index * model, std::array<uint8_t, 32> & result) -> bool {
        snapshot::sha256_hasher hash;
        if (!model) {
            static constexpr char absent[] = "absent";
            hash.update(absent, sizeof(absent) - 1);
            result = hash.finish();
            return true;
        }

        const auto feed_u64 = [&hash](uint64_t value) {
            uint8_t bytes[8];
            for (int i = 0; i < 8; ++i) bytes[i] = static_cast<uint8_t>(value >> (i * 8));
            hash.update(bytes, sizeof(bytes));
        };
        const auto feed_string = [&hash, &feed_u64](const std::string & value) {
            feed_u64(value.size());
            hash.update(value.data(), value.size());
        };

        feed_string(model->arch);
        for (const std::string & raw_path : model->shard_paths()) {
            std::error_code ec;
            const std::string path = std::filesystem::weakly_canonical(raw_path, ec).string();
            if (ec) return false;
            struct stat st {};
            if (::stat(path.c_str(), &st) != 0) return false;
            feed_string(path);
            feed_u64(static_cast<uint64_t>(st.st_dev));
            feed_u64(static_cast<uint64_t>(st.st_ino));
            feed_u64(static_cast<uint64_t>(st.st_size));
            feed_u64(static_cast<uint64_t>(st.st_mtime));
            feed_u64(static_cast<uint64_t>(st.st_ctime));
#if defined(__APPLE__)
            feed_u64(static_cast<uint64_t>(st.st_mtimespec.tv_nsec));
            feed_u64(static_cast<uint64_t>(st.st_ctimespec.tv_nsec));
#else
            feed_u64(static_cast<uint64_t>(st.st_mtim.tv_nsec));
            feed_u64(static_cast<uint64_t>(st.st_ctim.tv_nsec));
#endif
            std::ifstream input(path, std::ios::binary);
            if (!input) return false;
            constexpr size_t sample_size = 64u << 10;
            std::vector<uint8_t> sample(std::min<uint64_t>(sample_size, static_cast<uint64_t>(st.st_size)));
            if (!sample.empty()) {
                input.read(reinterpret_cast<char *>(sample.data()), static_cast<std::streamsize>(sample.size()));
                if (!input) return false;
                hash.update(sample.data(), sample.size());
                if (static_cast<uint64_t>(st.st_size) > sample.size()) {
                    input.clear();
                    input.seekg(static_cast<std::streamoff>(st.st_size - static_cast<off_t>(sample.size())));
                    input.read(reinterpret_cast<char *>(sample.data()), static_cast<std::streamsize>(sample.size()));
                    if (!input) return false;
                    hash.update(sample.data(), sample.size());
                }
            }
        }

        std::vector<const tensor_ref *> tensors;
        tensors.reserve(model->tensors().size());
        for (const auto & entry : model->tensors()) tensors.push_back(&entry.second);
        std::sort(tensors.begin(), tensors.end(), [](const tensor_ref * a, const tensor_ref * b) {
            return a->name < b->name;
        });
        feed_u64(tensors.size());
        for (const tensor_ref * tensor : tensors) {
            feed_string(tensor->name);
            feed_u64(static_cast<uint64_t>(tensor->shard));
            feed_u64(tensor->file_offset);
            feed_u64(static_cast<uint64_t>(tensor->type));
            for (const int64_t dim : tensor->ne) feed_u64(static_cast<uint64_t>(dim));
            feed_u64(tensor->nbytes);
        }
        result = hash.finish();
        return true;
    };

    std::array<uint8_t, 32> hot_digest{}, cold_digest{};
    if (!model_fingerprint(mi_, hot_digest) || !model_fingerprint(mi_cold_, cold_digest)) return {};
    append_string(out, digest_hex(hot_digest));
    append_string(out, digest_hex(cold_digest));

    // State shape and semantics. Performance-only cache/tiering choices are not
    // included; state placement and tensor types are.
    append_u32(out, cfg_.n_ctx);
    append_u32(out, cfg_.use_qsa);
    append_u32(out, cfg_.idx_host);
    append_u32(out, cfg_.kv_host);
    append_u32(out, cfg_.skip_miss);
    append_float(out, cfg_.gate_drop);
    append_u32(out, cfg_.indexer_top_k);
    append_u32(out, static_cast<uint32_t>(cfg_.type_k));
    append_u32(out, static_cast<uint32_t>(cfg_.type_v));

    append_u32(out, hp_.n_layer); append_u32(out, hp_.n_embd); append_u32(out, hp_.n_ctx_train);
    append_u32(out, hp_.n_vocab); append_u32(out, hp_.n_head); append_u32(out, hp_.n_head_kv);
    append_u32(out, hp_.n_embd_head_k); append_u32(out, hp_.n_embd_head_v); append_float(out, hp_.rms_eps);
    append_u32(out, hp_.full_attention_interval); append_vector32(out, hp_.compress_ratios);
    append_float(out, hp_.rope_freq_base); append_u32(out, hp_.rope_dim);
    for (const int32_t section : hp_.mrope_sections) append_i32(out, section);
    append_u32(out, hp_.n_expert); append_u32(out, hp_.n_expert_used);
    append_u32(out, hp_.n_ff_exp); append_u32(out, hp_.n_ff_shexp);
    append_u32(out, hp_.ssm_d_conv); append_u32(out, hp_.ssm_d_state);
    append_u32(out, hp_.ssm_n_group); append_u32(out, hp_.ssm_dt_rank); append_u32(out, hp_.ssm_d_inner);
    append_u32(out, hp_.idx_n_head); append_u32(out, hp_.idx_key_len); append_u32(out, hp_.idx_top_k);
    append_u32(out, hp_.hc_count); append_u32(out, hp_.hc_low_rank);
    append_vector32(out, hp_.ple_layers); append_u32(out, hp_.ple_ngram_size);
    append_u32(out, hp_.ple_heads_per_ngram); append_u32(out, hp_.ple_conv_kernel); append_u32(out, hp_.d_ple);
    append_vector64(out, hp_.ple_head_offsets); append_vector64(out, hp_.ple_head_vocab_sizes);
    append_vector64(out, hp_.ple_layer_multipliers); append_i32(out, hp_.ple_eos_token_id);
    append_i32(out, hp_.tok_bos); append_i32(out, hp_.tok_eos);
    append_i32(out, hp_.tok_pad); append_i32(out, hp_.tok_image);
    return out;
}

bool engine::save_snapshot(const std::string & path,
                           const std::vector<int32_t> & tokens,
                           const std::string & server_state,
                           uint64_t & bytes_written,
                           std::string & err) const {
    bytes_written = 0;
    if (!cfg_.mtp_path.empty()) {
        err = "snapshot v1 is incompatible with MTP";
        return false;
    }
    if (n_past_ <= 0 || tokens.size() != static_cast<size_t>(n_past_)) {
        err = "snapshot token history does not match engine n_past";
        return false;
    }
    const std::vector<uint8_t> compatibility = snapshot_compatibility();
    if (compatibility.empty()) {
        err = "could not fingerprint the loaded model";
        return false;
    }

    auto token_bytes = std::make_shared<const std::vector<uint8_t>>(encode_tokens(tokens));
    auto server_bytes = std::make_shared<const std::vector<uint8_t>>(server_state.begin(), server_state.end());
    std::vector<snapshot::section_source> sources;
    sources.reserve(2 + 3 * hp_.n_layer);
    sources.push_back(byte_source(SEC_TOKENS, 0, std::move(token_bytes)));
    sources.push_back(byte_source(SEC_SERVER_STATE, 0, std::move(server_bytes)));

    const uint64_t k_width = static_cast<uint64_t>(hp_.n_head_kv) * hp_.n_embd_head_k;
    const uint64_t v_width = static_cast<uint64_t>(hp_.n_head_kv) * hp_.n_embd_head_v;
    for (uint32_t layer = 0; layer < hp_.n_layer; ++layer) {
        if (ggml_tensor * k = st_.k_cache(layer)) {
            ggml_tensor * v = st_.v_cache(layer);
            ggml_tensor * idx = st_.idx_cache(layer);
            sources.push_back(tensor_source(SEC_K_CACHE, layer, k, prefix_bytes(k, k_width, n_past_)));
            sources.push_back(tensor_source(SEC_V_CACHE, layer, v, prefix_bytes(v, v_width, n_past_)));
            sources.push_back(tensor_source(SEC_INDEX_CACHE, layer, idx,
                                             prefix_bytes(idx, hp_.idx_key_len, n_past_)));
        } else {
            ggml_tensor * recurrent = st_.rs_state(layer);
            ggml_tensor * conv = st_.rs_conv(layer);
            if (!recurrent || !conv) {
                err = "snapshot state topology is incomplete at layer " + std::to_string(layer);
                return false;
            }
            sources.push_back(tensor_source(SEC_RECURRENT, layer, recurrent, ggml_nbytes(recurrent)));
            sources.push_back(tensor_source(SEC_CONV_HISTORY, layer, conv, ggml_nbytes(conv)));
        }
    }
    if (ggml_tensor * ple = st_.ple_conv())
        sources.push_back(tensor_source(SEC_PLE_HISTORY, 0, ple, ggml_nbytes(ple)));

    if (!snapshot::write_atomic(path, compatibility, sources, err)) return false;
    std::error_code ec;
    bytes_written = std::filesystem::file_size(path, ec);
    if (ec) {
        err = "snapshot was written but its size could not be read: " + ec.message();
        return false;
    }
    return true;
}

bool engine::restore_snapshot(const std::string & path,
                              const snapshot_metadata_validator & validate_metadata,
                              std::vector<int32_t> & tokens,
                              std::string & server_state,
                              uint64_t & bytes_read,
                              std::string & err) {
    tokens.clear();
    server_state.clear();
    bytes_read = 0;
    if (!cfg_.mtp_path.empty()) {
        err = "snapshot v1 is incompatible with MTP";
        return false;
    }

    snapshot::archive_index archive;
    if (!snapshot::inspect_and_validate(path, archive, err)) return false;
    bytes_read = archive.file_size;
    const std::vector<uint8_t> compatibility = snapshot_compatibility();
    if (compatibility.empty()) {
        err = "could not fingerprint the loaded model";
        return false;
    }
    if (archive.compatibility != compatibility) {
        err = "snapshot is incompatible with the running model or state layout";
        return false;
    }

    std::map<key, const snapshot::section_info *> sections;
    for (const auto & section : archive.sections) sections[{section.kind, section.layer}] = &section;
    const auto token_it = sections.find({SEC_TOKENS, 0});
    const auto server_it = sections.find({SEC_SERVER_STATE, 0});
    if (token_it == sections.end() || server_it == sections.end()) {
        err = "snapshot is missing server metadata sections";
        return false;
    }
    if ((token_it->second->length % 4) != 0 || token_it->second->length == 0 ||
        token_it->second->length / 4 > cfg_.n_ctx) {
        err = "snapshot token history has an invalid length";
        return false;
    }
    if (server_it->second->length > (256ull << 20)) {
        err = "snapshot server metadata exceeds 256 MiB";
        return false;
    }

    std::vector<uint8_t> token_bytes;
    std::vector<uint8_t> server_bytes;
    if (!read_section(path, *token_it->second, token_bytes, err) ||
        !read_section(path, *server_it->second, server_bytes, err)) return false;
    tokens.resize(token_bytes.size() / 4);
    for (size_t i = 0; i < tokens.size(); ++i) {
        const uint8_t * p = token_bytes.data() + i * 4;
        tokens[i] = static_cast<int32_t>(static_cast<uint32_t>(p[0]) |
                    (static_cast<uint32_t>(p[1]) << 8) |
                    (static_cast<uint32_t>(p[2]) << 16) |
                    (static_cast<uint32_t>(p[3]) << 24));
    }
    server_state.assign(server_bytes.begin(), server_bytes.end());
    if (validate_metadata && !validate_metadata(tokens, server_state, err)) {
        tokens.clear(); server_state.clear();
        return false;
    }

    const uint64_t n_past = tokens.size();
    const uint64_t k_width = static_cast<uint64_t>(hp_.n_head_kv) * hp_.n_embd_head_k;
    const uint64_t v_width = static_cast<uint64_t>(hp_.n_head_kv) * hp_.n_embd_head_v;
    std::map<key, std::pair<ggml_tensor *, uint64_t>> destinations;
    for (uint32_t layer = 0; layer < hp_.n_layer; ++layer) {
        if (ggml_tensor * k = st_.k_cache(layer)) {
            ggml_tensor * v = st_.v_cache(layer);
            ggml_tensor * idx = st_.idx_cache(layer);
            destinations[{SEC_K_CACHE, layer}] = {k, prefix_bytes(k, k_width, n_past)};
            destinations[{SEC_V_CACHE, layer}] = {v, prefix_bytes(v, v_width, n_past)};
            destinations[{SEC_INDEX_CACHE, layer}] = {idx, prefix_bytes(idx, hp_.idx_key_len, n_past)};
        } else {
            ggml_tensor * recurrent = st_.rs_state(layer);
            ggml_tensor * conv = st_.rs_conv(layer);
            if (!recurrent || !conv) {
                err = "snapshot state topology is incomplete at layer " + std::to_string(layer);
                tokens.clear(); server_state.clear();
                return false;
            }
            destinations[{SEC_RECURRENT, layer}] = {recurrent, ggml_nbytes(recurrent)};
            destinations[{SEC_CONV_HISTORY, layer}] = {conv, ggml_nbytes(conv)};
        }
    }
    if (ggml_tensor * ple = st_.ple_conv())
        destinations[{SEC_PLE_HISTORY, 0}] = {ple, ggml_nbytes(ple)};

    if (archive.sections.size() != destinations.size() + 2) {
        err = "snapshot section topology does not match the running engine";
        tokens.clear(); server_state.clear();
        return false;
    }
    for (const auto & [identity, destination] : destinations) {
        const auto it = sections.find(identity);
        if (it == sections.end() || it->second->length != destination.second) {
            err = "snapshot section is missing or has the wrong size (kind=" +
                  std::to_string(identity.first) + ", layer=" + std::to_string(identity.second) + ")";
            tokens.clear(); server_state.clear();
            return false;
        }
    }
    for (const auto & [identity, section] : sections) {
        if (identity != key{SEC_TOKENS, 0} && identity != key{SEC_SERVER_STATE, 0} &&
            destinations.find(identity) == destinations.end()) {
            err = "snapshot contains an unknown engine section";
            tokens.clear(); server_state.clear();
            return false;
        }
    }

    if (!snapshot::apply(path, archive,
        [&destinations](const snapshot::section_info & section,
                        snapshot::write_callback & sink, std::string & factory_err) {
            const auto it = destinations.find({section.kind, section.layer});
            if (it == destinations.end()) {
                sink = {};
                return true;
            }
            ggml_tensor * tensor = it->second.first;
            const uint64_t length = it->second.second;
            sink = [tensor, length](uint64_t offset, const void * src, size_t size, std::string & sink_err) {
                if (offset > length || size > length - offset) {
                    sink_err = "snapshot tensor destination range";
                    return false;
                }
                ggml_backend_tensor_set(tensor, src, static_cast<size_t>(offset), size);
                return true;
            };
            (void) factory_err;
            return true;
        }, err)) {
        // apply can discover an I/O race or checksum failure after partial device
        // copies. Never expose that state as a usable prefix.
        reset();
        clear_embeddings();
        tokens.clear(); server_state.clear();
        return false;
    }

    n_past_ = static_cast<int32_t>(n_past);
    mtp_have_h_ = false;
    mtp_kv_valid_ = true;
    mtp_draft_ = -1;
    rb_valid_ = false;
    rb_depth_ = 0;
    pool_dirty_ = true;
    clear_embeddings();
    if (qbuf_) {
        const int64_t count = qd_.bias->ne[0];
        std::vector<float> ninf(count, -std::numeric_limits<float>::infinity());
        ggml_backend_tensor_set(qd_.bias, ninf.data(), 0, ninf.size() * sizeof(float));
    }
    return true;
}

} // namespace qwfn
