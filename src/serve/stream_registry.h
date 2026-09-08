#pragma once

// StreamRegistry -- the serving-owned registry of in-flight streaming chat completions
// (llama-parity real-design §1, implementation step 1). It is the first real server-side
// mutable state added by the parity work: an in-flight generation's buffered SSE bytes and its
// owner-addressable cancel token outlive the rendering thread's local scope and stay addressable
// by X-Conversation-Id / completion id from *other* requests (the webui's lookup / resume /
// cancel routes).
//
// Keyed by the X-Conversation-Id header value (conversationId::model, the webui's stream
// identity). Each streaming chat completion is registered; its raw SSE byte stream is buffered
// so a client can resume from a byte offset (?from=N) and tail the stream live. Each entry also
// carries a std::shared_ptr<std::atomic<bool>> cancel token that the streaming handler OR-composes
// into the EXISTING CancellationView (include/ninfer/engine.h:49, types.h:616) so an owner can
// stop its own in-flight generation. This is owner-initiated stop, NOT scheduler preemption: it
// sets a flag on one request's token, reuses the existing cooperative-stop channel, and adds no
// Engine/Core mechanism (AGENTS.md ownership: serving-owned protocol state, no boundary crossing).
//
// The registry only OBSERVES the SSE bytes the serving handler already renders (it tees them);
// it does not schedule or run any generation.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ninfer::serve {

// One in-flight (or recently completed) streaming chat completion. The registry holds the entry
// (so its buffered bytes + cancel token outlive the rendering thread's local scope); the
// rendering handler also holds a shared_ptr for the generation's lifetime. Non-copyable (it owns
// a mutex/condvar); shared via std::shared_ptr.
struct StreamEntry {
    // The registry key (the X-Conversation-Id value = conversationId::model).
    std::string id;
    // The NInfer-generated chat completion id ("chatcmpl-...") for this generation.
    std::string completion_id;
    // Epoch seconds at which the stream was created (the webui's started_at ordering field).
    std::int64_t started_at = 0;

    // Set when the stream ends (success / error / cancel); records the retention start for reap.
    std::atomic<bool> is_done{false};
    // Owner-addressable cancel token: shared with the in-flight generation's CancellationView and
    // the DELETE/control handlers. Copying the shared_ptr is the ownership transfer (no race).
    std::shared_ptr<std::atomic<bool>> cancel;

    // --- Byte replay + live tail (guarded by buf_mu) ---
    // Tee one rendered SSE frame into the replay buffer (the exact bytes written to the live
    // client) and wake live tailers. The buffer holds the raw byte sequence so ?from=N replays a
    // true byte prefix/suffix (design §0.3, §1.1).
    void append(std::string_view frame);
    // Mark the stream ended (record the steady done time) and wake live tailers. Idempotent.
    void mark_done();
    // The buffered bytes from `from` (a byte offset); `from >= size` yields an empty string.
    [[nodiscard]] std::string replay_from(std::size_t from) const;
    // Total buffered bytes (the current byte offset a fresh tailer would resume from).
    [[nodiscard]] std::size_t buffered_bytes() const;
    // Block until the buffer grows past `offset` or the stream is done; return whether it is done.
    bool wait_for_bytes(std::size_t offset) const;
    // Whether the buffer hit its cap (a mid-stream resume degrades to a fresh attach when true).
    [[nodiscard]] bool truncated() const;
    // The steady-clock time the stream was marked done (starts the retention window for reap).
    [[nodiscard]] std::chrono::steady_clock::time_point done_at() const;

    StreamEntry() = default;
    StreamEntry(const StreamEntry&)            = delete;
    StreamEntry& operator=(const StreamEntry&) = delete;

private:
    // Per-stream buffer cap (design §1.2, "bounded memory"): a very long generation stops being
    // buffered once it exceeds this; a resume then degrades to a fresh attach (truncated==true).
    static constexpr std::size_t kMaxBufferSize = 4 * 1024 * 1024; // 4 MiB
    mutable std::mutex buf_mu;
    mutable std::condition_variable buf_cv;
    std::string buffer;
    std::chrono::steady_clock::time_point done_at_{};
    bool truncated_ = false;
};

// A record for a live (non-done) stream, in the shape the webui's selectActiveStream reads
// (conversation_id = the full stream id; is_done; started_at = epoch seconds, comparable).
struct LookupRecord {
    std::string conversation_id;
    bool is_done    = false;
    std::int64_t started_at = 0;
};

// The registry: create / get / lookup_active / remove / reap + the cancel-token accessors.
// Thread-safe: map access under mu_; per-entry buffer + done flag under buf_mu / is_done.
class StreamRegistry {
public:
    // Done entries are kept for `retention` after is_done, then reaped (design §1.2). A 60 s
    // default matches the router's TTL-tick cadence; a client that resumed past the end can do so
    // comfortably. Tests use a short window.
    explicit StreamRegistry(std::chrono::seconds retention = kDefaultRetention) noexcept;

    // Create + register a stream for a streaming chat completion. Returns the handle the handler
    // keeps for the generation's lifetime (the registry also holds one, until reaped). A new
    // generation supersedes a prior (re)entry for the same conversation id.
    [[nodiscard]] std::shared_ptr<StreamEntry> create(std::string id, std::string completion_id,
                                                     std::int64_t started_at);
    // Fetch an entry by stream id (null if unknown / reaped).
    [[nodiscard]] std::shared_ptr<StreamEntry> get(std::string_view id) const;
    // All live (non-done) streams, for POST /v1/streams/lookup (the handler filters the result by
    // the requested conversation_ids).
    [[nodiscard]] std::vector<LookupRecord> lookup_active() const;
    // Remove an entry by stream id (used by the reaper and DELETE). Returns true if found.
    [[nodiscard]] bool remove(std::string_view id);
    // Set the stream's cancel flag (owner stop; observed by the in-flight generation's
    // CancellationView). Returns true if the stream was found.
    [[nodiscard]] bool cancel(std::string_view id);
    // Set the cancel flag for the stream owning a completion id (the control endpoint's channel).
    // Returns true if found.
    [[nodiscard]] bool cancel_by_completion(std::string_view completion_id);
    // Reap done entries whose retention window has elapsed (now - done_at >= retention). Returns
    // the number removed. Only ever touches *done* entries; a live-tailer may hold a shared_ptr
    // to one concurrently (the shared_ptr keeps it alive).
    [[nodiscard]] std::size_t reap(std::chrono::steady_clock::time_point now);

    static constexpr std::chrono::seconds kDefaultRetention{60};

private:
    mutable std::mutex mu_; // locked by both the const lookups (get / lookup_active) and mutators
    std::unordered_map<std::string, std::shared_ptr<StreamEntry>> entries_;          // by stream id
    std::unordered_map<std::string, std::shared_ptr<StreamEntry>> completion_index_; // by completion id
    std::chrono::seconds retention_;
};

} // namespace ninfer::serve
