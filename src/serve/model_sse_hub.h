#pragma once

#include "serve/model_router.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace ninfer::serve {

// ModelSseHub -- the serving-owned model-status feed that drives GET /models/sse (llama-parity
// design §5, the router-hooked pub/sub tier). It is a small serving-side state object owned by
// HttpServer:
//   - it READS the current model status off the in-process router (loaded_id() / is_swapping()) to
//     build the "current status" record a new connection receives, AND
//   - it is the fan-out target of the router's status hook (ModelRouter::set_status_hook, wired in
//     HttpServer::attach): on every status transition (load/swap/unload/TTL-evict) the router
//     fires a ModelStatusEvent, and this hub fans it out to all connected subscribers.
//
// The fan-out is NON-BLOCKING with respect to the router: each subscriber owns a bounded queue,
// and publish() only enqueues (O(1) per subscriber). A slow client whose queue overflows is
// DROPPED (its subscriber is marked inactive) rather than stalling the router's event path --
// exactly the "drop-on-slow-client" policy the design calls for (design §5.1). The connected
// client's SSE writer thread drains its own queue (pop_blocking), so a blocked socket never stalls
// publish(). Keepalive comments still hold idle connections open on the kKeepaliveInterval cadence.
//
// Ownership: entirely in serving (the hub + router are both src/serve/). No device allocation, no
// weight repacking, no preemption (AGENTS.md: serving owns protocol translation; the router owns
// load/eviction). The hub is a read-only observer of the router + a pub/sub bus for its status
// events -- nothing more.
class ModelSseHub final {
public:
    // The keepalive cadence for a held-open /models/sse connection (the SseTransport heartbeat
    // interval). Matches the existing SSE heartbeat default (SseTransport::kHeartbeatInterval) so
    // the /models/sse stream reuses the same keepalive mechanism as the chat-completions stream; a
    // no-op SSE comment on this cadence keeps the connection live for the webui's probe.
    static constexpr std::chrono::seconds kKeepaliveInterval{5};

    // A /models/sse subscriber (one per connected client).
    using SubId = std::uint64_t;

    // Default per-subscriber queue depth. A subscriber that cannot keep up (its queue reaches this
    // depth while the router keeps firing events) is dropped by publish() rather than stalling the
    // fan-out (design §5.1). 64 events is far more than the webui's reconnect loop needs; the webui
    // treats a dropped (closed) stream as non-fatal and reattaches (design §0.8).
    static constexpr std::size_t kDefaultQueueDepth = 64;

    // Bind the router whose live status this hub reads. Called once by HttpServer::attach(); the
    // hub reads the router at record-build time (no copy of router state is held), so a null router
    // (before attach) yields the "unloaded" snapshot.
    void bind(ModelRouter& router);

    // The `data: ...` record a new connection receives on connect: the current model status in the
    // webui's wire format (event "model_status"; data.status in the Yo vocabulary, design §3.3).
    // The fan-out record (publish) is built by the SAME helper, so the two shapes are identical.
    [[nodiscard]] std::string current_status_record() const;

    // The keepalive cadence for a held-open connection, as the SseTransport heartbeat interval.
    [[nodiscard]] std::chrono::steady_clock::duration keepalive_interval() const {
        return kKeepaliveInterval;
    }

    // --- pub/sub (design §5.1) ---

    // Register a subscriber; returns its id. The hub pushes every published status record onto this
    // subscriber's bounded queue; the connection's SSE writer thread drains it via pop_blocking.
    // A subscriber whose queue overflows (a slow client) is dropped by publish() and its subsequent
    // pop_blocking returns Inactive. `queue_depth` bounds the queue (the default is plenty; a
    // host test shrinks it to exercise the drop path deterministically).
    SubId subscribe(std::size_t queue_depth = kDefaultQueueDepth);

    // Result of a blocking drain: a record was delivered (Got), the queue was idle for `timeout`
    // (TimedOut -> the writer emits a keepalive comment and loops), or the subscriber is gone
    // (Inactive -> it was dropped for being slow or unsubscribed, so the writer stops).
    enum class PopResult { Got, TimedOut, Inactive };

    // Drain one pending record, waiting up to `timeout` (the keepalive cadence). The connection's
    // SSE writer thread calls this in a loop: on Got it writes the record; on TimedOut it writes a
    // keepalive comment; on Inactive it stops (the subscriber was dropped or unsubscribed).
    PopResult pop_blocking(SubId id, std::string* out,
                           std::chrono::steady_clock::duration timeout);

    // Remove a subscriber (clean teardown on disconnect). Idempotent. A dropped (slow) subscriber
    // is also removed here when its writer finishes; the entry is the shared_ptr so an in-flight
    // publish() snapshot still holds it safely.
    void unsubscribe(SubId id);

    // Fan a status-change event to ALL connected subscribers (called by the router's status hook,
    // OUTSIDE the router lock). Formats the record ONCE (the exact wire shape of
    // current_status_record) and enqueues it on each subscriber; a subscriber whose queue overflows
    // is dropped (never blocks the fan-out). This is the non-blocking decoupling: the router hook
    // thread only enqueues; the per-connection writer thread does the (potentially blocking) socket
    // write.
    void publish(const ModelStatusEvent& ev);

    // True if `id` is still a registered subscriber (the writer loop uses it to know when to stop).
    [[nodiscard]] bool active(SubId id) const;

private:
    // One connected client's bounded queue. publish() enqueues under `mu`; the connection's writer
    // drains via pop_blocking. `dropped` is set when the queue overflows (a slow client) -- the
    // writer then stops, and the fan-out never blocks on this client.
    struct Subscriber {
        std::deque<std::string> queue;
        std::size_t depth;
        std::mutex mu; // guards queue + dropped (never held across a socket write)
        std::condition_variable cv;
        bool dropped = false;
    };

    // The bounded queue for `id`; nullptr once it has been unsubscribed (the shared_ptr copy in an
    // in-flight publish() snapshot keeps the Subscriber alive until the snapshot releases it).
    [[nodiscard]] std::shared_ptr<Subscriber> subscriber(SubId id) const;

    ModelRouter* router_ = nullptr;
    mutable std::mutex mu_; // guards subs_ + next_id_ (mutable: the const read hooks lock it)
    std::unordered_map<SubId, std::shared_ptr<Subscriber>> subs_;
    SubId next_id_ = 1; // monotonically increasing; assigned only under mu_
};

} // namespace ninfer::serve
