#include "serve/stream_registry.h"

#include <utility>

namespace ninfer::serve {

// --- StreamEntry -------------------------------------------------------------

void StreamEntry::append(std::string_view frame) {
    {
        std::lock_guard lock(buf_mu);
        if (truncated_) { return; }
        if (buffer.size() + frame.size() > kMaxBufferSize) {
            // Bounded memory (design §1.2): stop buffering; a mid-stream resume degrades to a
            // fresh attach (truncated()==true).
            truncated_ = true;
            return;
        }
        buffer.append(frame);
    }
    buf_cv.notify_all();
}

void StreamEntry::mark_done() {
    {
        std::lock_guard lock(buf_mu);
        done_at_ = std::chrono::steady_clock::now();
    }
    is_done.store(true, std::memory_order_release);
    buf_cv.notify_all();
}

std::string StreamEntry::replay_from(std::size_t from) const {
    std::lock_guard lock(buf_mu);
    if (from >= buffer.size()) { return {}; }
    return buffer.substr(from);
}

std::size_t StreamEntry::buffered_bytes() const {
    std::lock_guard lock(buf_mu);
    return buffer.size();
}

bool StreamEntry::wait_for_bytes(std::size_t offset) const {
    std::unique_lock lock(buf_mu);
    buf_cv.wait(lock, [this, offset] {
        return is_done.load(std::memory_order_acquire) || buffer.size() > offset;
    });
    return is_done.load(std::memory_order_acquire);
}

bool StreamEntry::truncated() const {
    std::lock_guard lock(buf_mu);
    return truncated_;
}

std::chrono::steady_clock::time_point StreamEntry::done_at() const {
    std::lock_guard lock(buf_mu);
    return done_at_;
}

// --- StreamRegistry ----------------------------------------------------------

StreamRegistry::StreamRegistry(std::chrono::seconds retention) noexcept : retention_(retention) {}

std::shared_ptr<StreamEntry> StreamRegistry::create(std::string id, std::string completion_id,
                                                    std::int64_t started_at) {
    auto entry = std::make_shared<StreamEntry>();
    entry->id            = std::move(id);
    entry->completion_id = std::move(completion_id);
    entry->started_at    = started_at;
    entry->cancel        = std::make_shared<std::atomic<bool>>(false);
    const auto shared = entry;
    {
        std::lock_guard lock(mu_);
        // A new generation supersedes a prior (re)entry for the same conversation id: drop the
        // stale completion mapping so cancel_by_completion resolves to the current generation.
        const auto existing = entries_.find(entry->id);
        if (existing != entries_.end()) { completion_index_.erase(existing->second->completion_id); }
        entries_.insert_or_assign(entry->id, shared);
        if (!entry->completion_id.empty()) { completion_index_.insert_or_assign(entry->completion_id, shared); }
    }
    return shared;
}

std::shared_ptr<StreamEntry> StreamRegistry::get(std::string_view id) const {
    std::lock_guard lock(mu_);
    const std::string key(id);
    const auto it = entries_.find(key);
    return it == entries_.end() ? nullptr : it->second;
}

std::vector<LookupRecord> StreamRegistry::lookup_active() const {
    std::lock_guard lock(mu_);
    std::vector<LookupRecord> records;
    records.reserve(entries_.size());
    for (const auto& [id, entry] : entries_) {
        if (entry->is_done.load(std::memory_order_acquire)) { continue; }
        records.push_back(LookupRecord{.conversation_id = id,
                                       .is_done         = false,
                                       .started_at      = entry->started_at});
    }
    return records;
}

bool StreamRegistry::remove(std::string_view id) {
    std::lock_guard lock(mu_);
    const std::string key(id);
    const auto it = entries_.find(key);
    if (it == entries_.end()) { return false; }
    completion_index_.erase(it->second->completion_id);
    entries_.erase(it);
    return true;
}

bool StreamRegistry::cancel(std::string_view id) {
    std::lock_guard lock(mu_);
    const std::string key(id);
    const auto it = entries_.find(key);
    if (it == entries_.end()) { return false; }
    it->second->cancel->store(true, std::memory_order_release);
    return true;
}

bool StreamRegistry::cancel_by_completion(std::string_view completion_id) {
    std::lock_guard lock(mu_);
    const std::string key(completion_id);
    const auto it = completion_index_.find(key);
    if (it == completion_index_.end()) { return false; }
    it->second->cancel->store(true, std::memory_order_release);
    return true;
}

std::size_t StreamRegistry::reap(std::chrono::steady_clock::time_point now) {
    std::lock_guard lock(mu_);
    std::size_t reaped = 0;
    for (auto it = entries_.begin(); it != entries_.end();) {
        auto& entry = it->second;
        // Only done entries are eligible; the shared_ptr in a live-tailer keeps one alive past
        // this erase. Holding mu_ here while locking the entry's buf_mu is safe: no path holds
        // buf_mu and then needs mu_ (append / mark_done / tail only take buf_mu).
        if (entry->is_done.load(std::memory_order_acquire) &&
            now - entry->done_at() >= retention_) {
            completion_index_.erase(entry->completion_id);
            it = entries_.erase(it);
            ++reaped;
        } else {
            ++it;
        }
    }
    return reaped;
}

} // namespace ninfer::serve
