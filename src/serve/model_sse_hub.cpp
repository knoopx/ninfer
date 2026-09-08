#include "serve/model_sse_hub.h"

#include <nlohmann/json.hpp>

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

// Derive the current status in the webui's Yo vocabulary from an ATOMIC router snapshot
// (design §3.3): a swap in flight -> "loading"; a loaded resident -> "loaded"; nothing loaded ->
// "unloaded". The snapshot is a single-lock read, so the status and the model id cannot diverge
// (a swap completing between two separate reads would otherwise pair "loading" with a stale id).
std::string_view status_from_snapshot(const RouterStatusSnapshot& snapshot) {
    if (snapshot.swapping) { return "loading"; }
    if (!snapshot.loaded_id.empty()) { return "loaded"; }
    return "unloaded";
}

// Build a model_status SSE record (the webui's wire format, design §3.3): a "data:" line, the JSON
// record, and an empty-line separator. `progress` is emitted only while loading (the webui reads it
// then); `exit_code` is 0 on success. BOTH current_status_record() (the connect record) and
// publish() (the fan-out record) use this, so the two shapes are byte-identical (the task's
// "keep the same record shape" requirement).
std::string build_status_record(std::string_view model, std::string_view status, double progress,
                                int exit_code) {
    nlohmann::json data = {{"status", status}, {"exit_code", exit_code}};
    if (status == "loading") { data["progress"] = progress; }
    const nlohmann::json record = {{"event", "model_status"}, {"model", model}, {"data", data}};
    // SSE wire format (design §3.3): a "data:" line, the JSON record, and an empty-line separator.
    return "data: " + record.dump() + "\n\n";
}

} // namespace

void ModelSseHub::bind(ModelRouter& router) { router_ = &router; }

std::string ModelSseHub::current_status_record() const {
    // The connect record: the current model status (a read-only snapshot off the router; null
    // router before attach yields the "unloaded" snapshot). Same shape as the fan-out record
    // (build_status_record). The snapshot is taken atomically (status + model id under one lock).
    const RouterStatusSnapshot snapshot =
        (router_ != nullptr) ? router_->status_snapshot() : RouterStatusSnapshot{};
    return build_status_record(snapshot.loaded_id, status_from_snapshot(snapshot),
                               /*progress=*/0.0, /*exit_code=*/0);
}

ModelSseHub::SubId ModelSseHub::subscribe(std::size_t queue_depth) {
    auto sub = std::make_shared<Subscriber>();
    sub->depth = (queue_depth == 0) ? kDefaultQueueDepth : queue_depth; // a zero depth is a no-op
    SubId id;
    {
        std::lock_guard lock(mu_);
        id = next_id_++;
        subs_[id] = sub;
    }
    return id;
}

std::shared_ptr<ModelSseHub::Subscriber> ModelSseHub::subscriber(SubId id) const {
    std::lock_guard lock(mu_);
    const auto it = subs_.find(id);
    return (it == subs_.end()) ? nullptr : it->second;
}

ModelSseHub::PopResult
ModelSseHub::pop_blocking(SubId id, std::string* out, std::chrono::steady_clock::duration timeout) {
    const auto sub = subscriber(id);
    if (sub == nullptr) {
        return PopResult::Inactive; // already unsubscribed (the writer stops)
    }
    std::unique_lock lock(sub->mu);
    // Wait for a record OR a drop (a slow client's queue overflow) OR the timeout (idle -> the
    // writer emits a keepalive comment and loops). `timeout` is the keepalive cadence.
    sub->cv.wait_for(lock, timeout, [&] { return !sub->queue.empty() || sub->dropped; });
    if (sub->dropped) {
        return PopResult::Inactive; // the client fell behind: stop (discard the backlog)
    }
    if (sub->queue.empty()) {
        return PopResult::TimedOut; // idle within the keepalive window
    }
    *out = std::move(sub->queue.front());
    sub->queue.pop_front();
    return PopResult::Got;
}

void ModelSseHub::unsubscribe(SubId id) {
    std::lock_guard lock(mu_);
    subs_.erase(id); // the shared_ptr may still be held by an in-flight publish()/pop_blocking; the
                     // Subscriber is destroyed when the last holder releases it (no leak)
}

void ModelSseHub::publish(const ModelStatusEvent& ev) {
    // Format the record ONCE (the exact wire shape of current_status_record), then fan it out to
    // every connected subscriber. Each subscriber's enqueue is O(1) (a bounded queue) and
    // NON-BLOCKING with respect to the socket write (the per-connection writer thread does that),
    // so a slow client never stalls the router's event path (design §5.1). A subscriber whose queue
    // overflows is dropped (marked inactive) rather than stalling the fan-out.
    const std::string record = build_status_record(ev.model, ev.status, ev.progress, ev.exit_code);

    // Snapshot the subscribers under the hub lock (bounded: one per connected client), then
    // enqueue each OUTSIDE the hub lock (the Subscriber has its own lock, so no lock cycle).
    std::vector<std::shared_ptr<Subscriber>> snapshot;
    {
        std::lock_guard lock(mu_);
        snapshot.reserve(subs_.size());
        for (const auto& entry : subs_) { snapshot.push_back(entry.second); }
    }
    for (const auto& sub : snapshot) {
        std::lock_guard lock(sub->mu);
        if (sub->dropped) {
            continue; // already dropped (an earlier overflow); nothing to enqueue
        }
        if (sub->queue.size() >= sub->depth) {
            sub->dropped = true;  // slow client: drop it (its writer wakes and stops)
            sub->cv.notify_all();
            continue;
        }
        sub->queue.push_back(record);
        sub->cv.notify_all();
    }
}

bool ModelSseHub::active(SubId id) const {
    std::lock_guard lock(mu_);
    return subs_.count(id) > 0;
}

} // namespace ninfer::serve
