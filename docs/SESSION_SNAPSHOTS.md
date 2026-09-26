# Durable session snapshots

**Status:** archive codec, engine capture/restore, and the single-slot HTTP contract
are implemented on `feature/session-snapshots`. CPU compilation and codec tests
pass. CUDA validation and production A/B are still required before merge.

## Purpose

QwFNfer owns one mutable inference state. Alternating long Hermes sessions
therefore destroys the active prefix and forces a full prefill on every switch.
A durable snapshot lets a session save that state before eviction and restore it
without running the model over the prefix again.

This format is QwFNfer-specific. It is not compatible with llama.cpp slot files,
and the decoder rejects any file whose magic, format version, integrity digest,
model fingerprint, context geometry, or section topology does not match.

## v1 scope and invariants

Version 1 snapshots exactly one idle server session. They contain:

- evaluated token IDs (`server::consumed`), which define `n_past`;
- the server continuation metadata (`consumed_img`, `last_gen`, `last_prompt`,
  `last_msgs`, content/reasoning/tool-call state, and thinking mode);
- the populated prefix of every attention-layer K, V, and sparse-index cache;
- every recurrent-layer DeltaNet state and short-convolution history;
- the PLE convolution history when present.

The derived QSA pool and bias are not serialized. Restore marks the pool dirty,
sets the bias tensor to negative infinity, and clears embedding overrides so the
next decode rebuilds derived data from the restored raw state.

Rollback state is not canonical session state and is invalidated after restore.
MTP/speculative state is not supported in v1: save and restore reject an engine
with an MTP head loaded. The deployed non-MTP `--spec-block` configuration is
supported; `--spec-block` only controls expert prefetching.

A snapshot is accepted only when all of these hold:

1. the archive validates completely before any engine tensor is changed;
2. its compatibility blob equals the running engine's blob;
3. token count is nonzero, within `n_ctx`, and equals every prefix section's
   expected byte length;
4. the section set is exact: no missing, duplicate, unknown, or extra section;
5. server metadata parses strictly and its prompt/generated-token continuity is
   consistent with the evaluated tokens;
6. the archive is unchanged when the second streaming pass copies tensors.

If a copy or second-pass integrity check fails, `engine::reset()` is called and
the server continuation metadata is cleared. A partial restore is never exposed
as a reusable session.

## Compatibility fingerprint

The compatibility blob includes format/layout identifiers, state-relevant engine
and model hyperparameters, context length, host/device state placement, KV types,
QSA and approximation settings, and SHA-256 deployment fingerprints for the hot
and optional cold model indices.

Each deployment fingerprint covers model architecture, sorted tensor metadata,
canonical shard paths, filesystem identity/timestamps/sizes, and samples from
the beginning and end of every shard. It is intentionally strict: replacing,
copying, or retiming model files invalidates old snapshots rather than risking a
restore into a different model.

## Archive format

All integers are little-endian. The fixed header is 64 bytes:

| Offset | Size | Field |
|---:|---:|---|
| 0 | 8 | magic `QWFNSN1\0` |
| 8 | 4 | format version (`1`) |
| 12 | 4 | compatibility-blob length |
| 16 | 4 | section count |
| 20 | 4 | reserved, must be zero |
| 24 | 8 | body length |
| 32 | 32 | SHA-256 of the entire body |

The body starts with the compatibility blob, followed by `section_count`
records. Each record has a 48-byte header and a streamed payload:

| Size | Field |
|---:|---|
| 4 | section kind |
| 4 | layer index |
| 8 | payload length |
| 32 | SHA-256 of this section payload |
| variable | payload bytes |

Section kinds are:

- `1`: consumed token IDs;
- `2`: JSON server-continuation metadata;
- `10`, `11`, `12`: K, V, and sparse-index prefix, keyed by layer;
- `20`, `21`: recurrent state and convolution history, keyed by layer;
- `30`: PLE convolution history.

Declared lengths are checked against file size with overflow-safe arithmetic.
The reader rejects truncation, trailing data, duplicate identities, unsupported
versions, a corrupt section digest, or a corrupt archive digest.

## Durable writes

`write_atomic()` streams source callbacks into a private temporary file in the
target directory, computes per-section and archive SHA-256 digests, writes the
final header, calls `fsync()` on the file, atomically renames it over the target,
and `fsync()`s the parent directory. Temporary and final files are mode `0600`.
Failures before rename remove the temporary file and leave an existing target
unchanged.

No full tensor payload is accumulated in RAM. Engine copies use bounded archive
chunks and `ggml_backend_tensor_get/set` with explicit byte offsets.

## HTTP contract

Snapshots are disabled unless the server starts with:

```text
--snapshot-dir /absolute/private/directory
```

The directory is created if needed and forced to mode `0700`. Filenames must be
ASCII basenames containing only letters, digits, `.`, `_`, or `-`; path
separators, dot entries, and symlinks are rejected.

The llama.cpp-compatible endpoints are:

```text
POST /slots/0?action=save
POST /slots/0?action=restore
Content-Type: application/json

{"filename":"local-llm-<key>.bin"}
```

Success responses include `id_slot`, `filename`, token count (`n_saved` or
`n_restored`), and archive bytes. `GET /slots` reports one slot with
`n_prompt_tokens`, allowing `local-llm-kv-cache` to verify the active prefix.
Only slot `0` exists.

Representative failures:

- `400`: malformed action/body/filename or unsupported slot ID;
- `404`: restore target absent or not a regular file;
- `409`: empty save, MTP enabled, model/config mismatch, or metadata mismatch;
- `422`: structurally invalid/corrupt archive;
- `500`: I/O or backend-copy failure.

Handlers take the same server mutex as inference. Snapshot work therefore cannot
race generation or another save/restore operation.

## Cache-proxy deployment contract

`local-llm-kv-cache` and QwFNfer must run on the same host and use the same
physical directory: the proxy performs the final temporary-to-stable filename
rename after the server save returns.

QwFNfer v1 deliberately does not expose llama.cpp `/tokenize` or
`/apply-template`, so configure the proxy as session-only:

```text
PI_LLAMA_CACHE_ENABLE_PREFIX_SEEDING=0
PI_LLAMA_CACHE_ENABLE_SHARED_PREFIXES=0
PI_LLAMA_CACHE_DIR=/var/lib/qwfnfer/snapshots
PI_LLAMA_CACHE_NAMESPACE=qwfnfer-<model-and-build-revision>
```

A dedicated namespace prevents old llama.cpp `.bin` files from being proposed
to QwFNfer. Even if one is presented, archive magic validation rejects it.

## Verification gates

Before production merge:

1. run the standalone codec suite, including round trip, replacement, mode,
   corruption, truncation, trailing data, bad magic/version, duplicate section,
   and changed-after-validation cases;
2. compile `qwfn-server` against the exact llama.cpp ABI on CPU and CUDA;
3. save and restore a real CUDA session, then compare continuation tokens/logits
   against an uninterrupted control;
4. exercise malformed filenames, corrupt files, compatibility mismatch, and MTP
   rejection;
5. run a two-session alternating A/B and compare prefill time, restore time,
   token continuity, and error/reset counters;
6. keep the previous service package and configuration as the rollback target.

The acceptance criterion is not merely a successful restore response: alternating
sessions must stop doing full historical prefill while exact-extension checks
continue to reset on any divergent prompt.
