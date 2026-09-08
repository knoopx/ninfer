// Host-side (no GPU) test for the serving-owned StreamRegistry (llama-parity real-design §1,
// step 1). Exercises the registry lifecycle, byte-offset replay, live tailing on the entry
// condvar, the owner cancel token, and the completion-index / cancel / remove accessors. No
// Engine, no model, no network -- a pure serving-state object. Bare main() + a small CHECK
// helper, matching the other host tests; the binary exits non-zero on any failure.

#include "serve/stream_registry.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace ns = ninfer::serve;
using ns::LookupRecord;
using ns::StreamEntry;
using ns::StreamRegistry;

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::fprintf(stderr, "FAIL: %s\n", message);
    return 1;
}

// create -> lookup(active) -> mark done -> excluded from active -> reaped after the retention
// window (a short window keeps the test from waiting a real 60 s).
int test_lifecycle() {
    int failures = 0;
    StreamRegistry registry(std::chrono::seconds(1));

    const auto entry = registry.create("conv-a::model", "chatcmpl-1", 100);
    failures += check(entry != nullptr && entry->id == "conv-a::model" &&
                          entry->completion_id == "chatcmpl-1" && entry->started_at == 100,
                      "create returned a bad entry");

    const auto active = registry.lookup_active();
    failures += check(active.size() == 1 && active[0].conversation_id == "conv-a::model" &&
                          !active[0].is_done && active[0].started_at == 100,
                      "lookup_active did not report the active stream");

    failures += check(registry.get("conv-a::model") == entry, "get returned a different entry");

    entry->mark_done();
    failures += check(entry->is_done.load(), "entry not done after mark_done");
    failures += check(registry.lookup_active().empty(), "done stream still reported active");

    // Within the retention window the done entry is kept; after it, it is reaped.
    const auto now = std::chrono::steady_clock::now();
    failures += check(registry.reap(now + std::chrono::milliseconds(400)) == 0,
                      "reap dropped a still-retained done stream");
    failures += check(registry.reap(now + std::chrono::milliseconds(1400)) == 1,
                      "reap did not drop the expired done stream");
    failures += check(registry.get("conv-a::model") == nullptr, "reaped stream is still present");
    return failures;
}

// A live (non-done) entry is NEVER reaped regardless of how far `now` is; only done + expired
// entries are removed.
int test_reap_keeps_active() {
    int failures = 0;
    StreamRegistry registry(std::chrono::seconds(1));
    const auto a = registry.create("conv-a::model", "chatcmpl-1", 100);
    const auto b = registry.create("conv-b::model", "chatcmpl-2", 101);
    auto now = std::chrono::steady_clock::now();
    failures += check(registry.reap(now + std::chrono::hours(1)) == 0,
                      "reap dropped an active stream");
    b->mark_done();
    now = std::chrono::steady_clock::now();
    failures += check(registry.reap(now + std::chrono::milliseconds(1400)) == 1,
                      "reap did not drop the single expired done stream");
    (void)a; // conv-a stays active for the whole test
    return failures;
}

// ?from=N replays the exact byte suffix from N (the webui's resume contract, design §0.3): the
// buffered raw SSE bytes, not a list of events.
int test_byte_offset_replay() {
    int failures = 0;
    StreamRegistry registry;
    const auto entry = registry.create("conv::model", "chatcmpl-1", 1);
    const std::string f1 = "data: one\n\n";
    const std::string f2 = "data: two\n\n";
    const std::string f3 = "data: [DONE]\n\n";
    entry->append(f1);
    entry->append(f2);
    entry->append(f3);

    const std::string all = f1 + f2 + f3;
    const std::size_t off2 = f1.size();
    failures += check(entry->replay_from(0) == all, "replay from 0 is not the full stream");
    failures += check(entry->replay_from(off2) == (f2 + f3), "byte-offset replay from N is wrong");
    failures += check(entry->replay_from(999999) == "", "replay past the end is not empty");
    failures += check(entry->buffered_bytes() == all.size(), "buffered byte count is wrong");
    return failures;
}

// A consumer that blocks on the entry condvar (wait_for_bytes) receives frames appended after it
// attached, delivering the buffered stream in order from its attach offset until done.
int test_live_tail() {
    int failures = 0;
    StreamRegistry registry;
    const auto entry = registry.create("conv::model", "chatcmpl-1", 1);
    const std::string f1 = "data: one\n\n";
    const std::string f2 = "data: two\n\n";
    const std::string f3 = "data: [DONE]\n\n";
    entry->append(f1); // buffered before the consumer attaches

    std::atomic<bool> blocked{false};
    std::atomic<bool> finished{false};
    std::vector<std::string> chunks;
    std::thread consumer([&] {
        std::size_t offset = 0;
        while (true) {
            const std::string have = entry->replay_from(offset);
            if (!have.empty()) {
                chunks.push_back(have);
                offset += have.size();
            }
            blocked = true;
            const bool done = entry->wait_for_bytes(offset);
            if (done) {
                const std::string tail = entry->replay_from(offset);
                if (!tail.empty()) { chunks.push_back(tail); }
                break;
            }
        }
        finished = true;
    });

    // Wait until the consumer has consumed f1 and is about to block on the condvar, then append
    // the next frames and finish. A bounded spin keeps this deterministic: the consumer always
    // reaches the blocked=true assignment on its first iteration.
    while (!blocked.load(std::memory_order_acquire)) { std::this_thread::yield(); }
    entry->append(f2);
    entry->append(f3);
    entry->mark_done();
    consumer.join();

    failures += check(finished.load(), "live-tail consumer did not finish");
    failures += check(chunks.size() == 2 && chunks[0] == f1 && chunks[1] == (f2 + f3),
                      "live tail did not deliver buffered frames in order from the attach offset");
    return failures;
}

// The owner cancel token is observable once set, and shared across handles (ownership transfer
// via the shared_ptr -- what the in-flight generation's CancellationView holds).
int test_cancel_token() {
    int failures = 0;
    StreamRegistry registry;
    const auto entry = registry.create("conv::model", "chatcmpl-1", 1);
    failures += check(entry->cancel != nullptr && !entry->cancel->load(),
                      "fresh entry's cancel token must be non-null and clear");
    entry->cancel->store(true, std::memory_order_release);
    failures += check(entry->cancel->load(), "cancel token not observable after being set");
    const auto alias = entry->cancel;
    failures += check(alias == entry->cancel && alias->load(), "cancel token not shared across handles");
    return failures;
}

// The registry's cancel accessors reach the same entry by stream id AND by completion id (the
// DELETE and control channels, design §2.2), and report a miss honestly.
int test_registry_cancel() {
    int failures = 0;
    StreamRegistry registry;
    const auto entry_a = registry.create("conv-a::model", "chatcmpl-1", 1);
    (void)entry_a; // registered so registry.cancel("conv-a::model") can find it
    failures += check(registry.cancel("conv-a::model"), "registry.cancel missed the stream");
    failures += check(!registry.cancel("conv-missing"), "cancel reported a missing stream");
    const auto entry2 = registry.create("conv-b::model", "chatcmpl-2", 2);
    failures += check(!entry2->cancel->load(), "entry2 cancel token not initially clear");
    failures += check(registry.cancel_by_completion("chatcmpl-2"),
                      "cancel_by_completion missed the stream");
    failures += check(entry2->cancel->load(), "cancel_by_completion did not reach the entry");
    failures += check(!registry.cancel_by_completion("chatcmpl-x"),
                      "cancel_by_completion reported a missing completion");
    return failures;
}

// remove() erases the entry (and its completion mapping) so a later get/cancel finds nothing.
int test_remove() {
    int failures = 0;
    StreamRegistry registry;
    const auto entry = registry.create("conv-a::model", "chatcmpl-1", 1);
    (void)entry; // registered so remove("conv-a::model") can find it
    failures += check(!registry.remove("conv-missing"), "remove reported a missing stream");
    failures += check(registry.remove("conv-a::model"), "remove did not find the stream");
    failures += check(registry.get("conv-a::model") == nullptr, "removed stream still present");
    failures += check(!registry.cancel_by_completion("chatcmpl-1"),
                      "removed stream's completion mapping survived");
    return failures;
}

// POST /v1/streams/lookup (handler step 3): the handler takes lookup_active() (all live, non-done
// entries) and filters to the requested conversation_ids (client-side filter; the registry returns
// all active and the handler narrows). An absent/empty request set returns all active (the
// design's selectActiveStream behavior).
int test_lookup_active_filter() {
    int failures = 0;
    StreamRegistry registry;
    const auto a = registry.create("conv-a::model", "chatcmpl-a", 100);
    const auto b = registry.create("conv-b::model", "chatcmpl-b", 101);
    const auto c = registry.create("conv-c::model", "chatcmpl-c", 102);
    c->mark_done(); // conv-c is done -> excluded from active
    (void)a;
    (void)b;

    const auto active = registry.lookup_active();
    failures += check(active.size() == 2, "lookup_active did not return exactly the live entries");

    // The handler's filter to a requested set (only conv-b requested) keeps only the match.
    const std::vector<std::string> requested{"conv-b::model"};
    std::vector<LookupRecord> filtered;
    for (const auto& record : active) {
        if (std::any_of(requested.begin(), requested.end(),
                        [&record](const std::string& id) { return record.conversation_id == id; })) {
            filtered.push_back(record);
        }
    }
    failures += check(filtered.size() == 1 && filtered[0].conversation_id == "conv-b::model" &&
                          !filtered[0].is_done && filtered[0].started_at == 101,
                      "lookup active-filter returned the wrong entries");

    // An empty request set returns all active (the handler returns the unfiltered active list).
    const std::vector<std::string> none;
    std::size_t all = 0;
    for (const auto& record : active) {
        if (none.empty() ||
            std::any_of(none.begin(), none.end(), [&record](const std::string& id) {
                return record.conversation_id == id;
            })) {
            ++all;
        }
    }
    failures += check(all == 2, "empty filter did not return all active entries");
    return failures;
}

// DELETE /v1/stream/{id} (handler step 5): raise the owner cancel token (the in-flight generation's
// CancellationView observes it) then remove the entry. A second DELETE on the now-unknown id is an
// honest no-op (the client must still get 200, not a 404).
int test_delete_cancel_remove() {
    int failures = 0;
    StreamRegistry registry;
    const auto entry = registry.create("conv::model", "chatcmpl-1", 1);
    // A stand-in for the in-flight generation's CancellationView holding the shared cancel token.
    const auto cancel = entry->cancel;
    failures += check(cancel != nullptr && !cancel->load(),
                      "fresh entry's cancel token must be clear");

    // The handler's DELETE flow: cancel(id) then remove(id).
    const bool found   = registry.cancel("conv::model");
    const bool removed = registry.remove("conv::model");
    failures += check(found, "DELETE did not find the in-flight stream");
    failures += check(removed, "DELETE did not remove the stream");
    failures += check(cancel->load(), "DELETE did not raise the shared cancel token");
    failures += check(registry.get("conv::model") == nullptr, "DELETE did not remove the entry");

    // A second DELETE on the now-unknown id is a no-op.
    failures += check(!registry.cancel("conv::model"), "second DELETE reported a missing stream");
    failures += check(!registry.remove("conv::model"),
                      "second DELETE remove reported a missing stream");
    (void)entry;
    return failures;
}

// POST /v1/chat/completions/control (handler step 6): resolve the completion id through the
// registry's completion index and raise that entry's cancel token (the in-flight generation's
// CancellationView observes it). An unknown completion id is an honest no-op (no token set,
// returns false -> the handler answers success:false).
int test_control_cancel_by_completion() {
    int failures = 0;
    StreamRegistry registry;
    const auto entry = registry.create("conv::model", "chatcmpl-xyz", 1);
    const auto cancel = entry->cancel;
    failures += check(!cancel->load(), "entry's cancel token must start clear");

    failures += check(registry.cancel_by_completion("chatcmpl-xyz"),
                      "control did not resolve the completion id");
    failures += check(cancel->load(), "control did not raise the entry's cancel token");

    failures += check(!registry.cancel_by_completion("chatcmpl-unknown"),
                      "control reported an unknown completion id");
    (void)entry;
    return failures;
}

// GET /v1/stream/{id}?from=N (handler step 4): the handler replays the buffered bytes from the
// offset, then live-tails new frames until done. The terminal "data: [DONE]" frame is appended by
// the producer (the terminal-frame loop tees it into the buffer), so the replayed/tailed bytes
// already END with [DONE] and the handler must NOT emit a second one. This drives the handler's
// exact loop (replay_from + wait_for_bytes) and asserts the emitted stream is the exact buffer
// suffix with a single terminal [DONE].
int test_get_replay_and_tail() {
    int failures = 0;
    StreamRegistry registry;
    const auto entry = registry.create("conv::model", "chatcmpl-1", 1);
    const std::string f1   = "data: one\n\n";
    const std::string f2   = "data: two\n\n";
    const std::string done = "data: [DONE]\n\n";
    entry->append(f1); // buffered before the consumer attaches (resume past f1)

    // A producer thread appends the remaining frames and marks done (as the real generation would).
    std::atomic<bool> finished{false};
    std::thread producer([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        entry->append(f2);
        entry->append(done); // the producer tees the terminal [DONE] into the buffer
        entry->mark_done();
        finished = true;
    });

    // The handler's loop, resuming from an offset into the middle of the stream (after f1).
    std::size_t offset = f1.size();
    std::string emitted;
    for (;;) {
        const std::string have = entry->replay_from(offset);
        if (!have.empty()) {
            emitted += have;
            offset += have.size();
        }
        const bool is_done = entry->wait_for_bytes(offset);
        if (is_done) {
            const std::string tail = entry->replay_from(offset);
            if (!tail.empty()) { emitted += tail; }
            break;
        }
    }
    producer.join();
    failures += check(finished.load(), "producer did not finish");
    failures += check(emitted == (f2 + done), "GET replay+tail did not emit the exact buffer suffix");
    // Double-emit safety: exactly ONE terminal [DONE] in the emitted stream (it came from the
    // buffer, not from the handler).
    std::size_t occurrences = 0;
    std::size_t pos         = 0;
    while ((pos = emitted.find("data: [DONE]", pos)) != std::string::npos) {
        ++occurrences;
        ++pos;
    }
    failures += check(occurrences == 1, "GET replay+tail must emit the terminal [DONE] exactly once");
    return failures;
}

} // namespace

int main() {
    int failures = test_lifecycle() + test_reap_keeps_active() + test_byte_offset_replay() +
                   test_live_tail() + test_cancel_token() + test_registry_cancel() + test_remove() +
                   test_lookup_active_filter() + test_delete_cancel_remove() +
                   test_control_cancel_by_completion() + test_get_replay_and_tail();
    if (failures == 0) { std::printf("ok\n"); }
    return failures == 0 ? 0 : 1;
}
