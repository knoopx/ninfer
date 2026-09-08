// In-process multi-model router implementation. See model_router.h for the grounding map to
// llama-swap (router/base.go, router/group.go, router/scheduler/fifo.go, router/loading.go,
// process/process.go) and the Option-B Engine-lifecycle adaptation.

#include "serve/model_router.h"

#include <chrono>
#include <thread>

namespace ninfer::serve {

// ---------------------------------------------------------------------------
// EngineModelBackend (Option-B production backend)
// ---------------------------------------------------------------------------

EngineModelBackend::EngineModelBackend(const ModelConfig& model, ServeOptions base_options,
                                       StartupObserver observer)
    : model_id_(model.id) {
    // Option-B: build the per-model backend by copying the base options and setting artifact_path =
    // model.artifact (the per-model input for a swap). The GenerationService ctor then constructs
    // the Engine from it. (Option-A's `startPort` is unused here — an Engine is in-process, not a
    // started child process.)
    ServeOptions per_model = base_options;
    per_model.artifact_path = model.artifact;
    // Apply per-model engine overrides (nullopt fields keep the global ServeOptions value).
    if (model.overrides.max_context)         per_model.max_context          = *model.overrides.max_context;
    if (model.overrides.default_max_tokens)  per_model.default_max_tokens   = *model.overrides.default_max_tokens;
    if (model.overrides.kv_capacity)         per_model.kv_capacity          = *model.overrides.kv_capacity;
    if (model.overrides.kv_cache)            per_model.kv_cache             = *model.overrides.kv_cache;
    if (model.overrides.speculative)         per_model.speculative          = *model.overrides.speculative;
    if (model.overrides.prefill_chunk)       per_model.prefill_chunk        = *model.overrides.prefill_chunk;
    if (model.overrides.enable_vision)       per_model.enable_vision        = *model.overrides.enable_vision;
    service_ = std::make_unique<GenerationService>(per_model, std::move(observer));
}

std::string EngineModelBackend::id() const { return model_id_; }
bool EngineModelBackend::is_available() const { return service_->is_available(); }

PreparedRequest
EngineModelBackend::prepare(const GenerationRequest& req, GenerationConsumerMode consumer_mode,
                            ninfer::GenerationObservationOptions observation,
                            std::function<bool()> is_cancelled, ContextCacheHints context_cache) const {
    return service_->prepare(req, consumer_mode, observation, std::move(is_cancelled),
                             context_cache);
}

int EngineModelBackend::count_prompt_tokens(const GenerationRequest& req,
                                            std::function<bool()> is_cancelled) const {
    return service_->count_prompt_tokens(req, std::move(is_cancelled));
}

GenerationOutcome EngineModelBackend::run(PreparedRequest& prepared, const StreamSink* sink,
                                          std::function<bool()> is_cancelled) {
    return service_->run(prepared, sink, std::move(is_cancelled));
}

ninfer::LoadSummary EngineModelBackend::load_summary() const { return service_->load_summary(); }
void EngineModelBackend::warmup() { service_->warmup(); }
ninfer::RuntimeStats EngineModelBackend::runtime_stats() const { return service_->runtime_stats(); }
ninfer::MemorySummary EngineModelBackend::memory_summary() const {
    return service_->memory_summary();
}
ninfer::ModelSamplingDefaults EngineModelBackend::sampling_defaults() const {
    return service_->sampling_defaults();
}
const ninfer::EngineOptions& EngineModelBackend::engine_options() const {
    return service_->engine_options();
}

const ServeOptions& EngineModelBackend::serve_options() const { return service_->options(); }

// ---------------------------------------------------------------------------
// GroupSwapper (llama-swap group.go groupSwapper + FindConfig)
// ---------------------------------------------------------------------------

GroupSwapper::GroupSwapper(const Config& config) : config_(config) {
    for (const auto& model : config_.models) {
        model_by_id_[model.id] = &model; // stable: config_.models is not reallocated after this
        for (const auto& alias : model.aliases) {
            alias_to_id_[alias] = model.id;
        }
    }
    for (const auto& [group_id, group] : config_.groups) {
        for (const auto& member : group.members) {
            model_to_group_[member] = group_id;
        }
    }
}

const ModelConfig* GroupSwapper::resolve(std::string_view id_or_alias) const {
    const std::string key(id_or_alias);
    if (const auto it = model_by_id_.find(key); it != model_by_id_.end()) {
        return it->second;
    }
    if (const auto alias_it = alias_to_id_.find(key); alias_it != alias_to_id_.end()) {
        const auto model_it = model_by_id_.find(alias_it->second);
        if (model_it != model_by_id_.end()) {
            return model_it->second;
        }
    }
    return nullptr;
}

std::string GroupSwapper::group_of(std::string_view model) const {
    const std::string key(model);
    if (const auto it = model_to_group_.find(key); it != model_to_group_.end()) {
        return it->second;
    }
    return key; // its own implicit exclusive group (single-resident semantics)
}

std::vector<std::string>
GroupSwapper::eviction_for(const std::string& target, const std::vector<std::string>& running) const {
    // ground: group.go groupSwapper.EvictionFor (the `consider` closure).
    const std::string target_group = group_of(target);
    const GroupConfig* target_cfg = group(target_group);
    const bool target_swap = target_cfg ? target_cfg->swap : true; // implicit group: swap
    const bool target_exclusive = target_cfg ? target_cfg->exclusive : true; // implicit: exclusive

    std::vector<std::string> result;
    for (const auto& model_id : running) {
        if (model_id == target) {
            continue;
        }
        const std::string other_group = group_of(model_id);
        const GroupConfig* other_cfg = group(other_group);
        const bool other_persistent = other_cfg ? other_cfg->persistent : false;
        if (other_group == target_group) {
            // Same-group sibling: evict when the group swaps.
            if (target_swap) {
                result.push_back(model_id);
            }
        } else if (target_exclusive) {
            // Cross-group: evict when the target's group is exclusive and the other group is not
            // persistent.
            if (!other_persistent) {
                result.push_back(model_id);
            }
        }
    }
    return result;
}

const GroupConfig* GroupSwapper::group(std::string_view group_id) const {
    const std::string key(group_id);
    const auto it = config_.groups.find(key);
    return it != config_.groups.end() ? &it->second : nullptr;
}

// ---------------------------------------------------------------------------
// FifoScheduler (llama-swap scheduler/fifo.go)
// ---------------------------------------------------------------------------

FifoScheduler::FifoScheduler(std::size_t max_queue_size, int limit)
    : max_queue_size_(max_queue_size), limit_(limit) {}

int FifoScheduler::in_flight(std::string_view model) const {
    const auto it = in_flight_.find(std::string(model));
    return it != in_flight_.end() ? it->second : 0;
}

bool FifoScheduler::admit(std::string_view model) const {
    return in_flight(model) < limit_;
}

bool FifoScheduler::enqueue(std::string_view model) {
    if (pending_.size() >= max_queue_size_) {
        return false; // bounded: drop when full (the router rejects the request)
    }
    pending_.push_back(std::string(model));
    return true;
}

std::optional<std::string> FifoScheduler::pop_front() {
    if (pending_.empty()) {
        return std::nullopt;
    }
    std::string front = pending_.front(); // copy, then remove (a returned reference would dangle)
    pending_.pop_front();
    return front;
}

int FifoScheduler::queue_position(std::string_view model) const {
    const std::string key(model);
    for (std::size_t i = 0; i < pending_.size(); ++i) {
        if (pending_[i] == key) {
            return static_cast<int>(i + 1); // 1-based
        }
    }
    return 0;
}

std::size_t FifoScheduler::queued() const { return pending_.size(); }

void FifoScheduler::begin_in_flight(std::string_view model) { ++in_flight_[std::string(model)]; }

void FifoScheduler::release_in_flight(std::string_view model) {
    const auto it = in_flight_.find(std::string(model));
    if (it == in_flight_.end() || it->second <= 0) {
        return;
    }
    if (--(it->second) == 0) {
        in_flight_.erase(it);
    }
}

int FifoScheduler::in_flight_total() const {
    int total = 0;
    for (const auto& [_, count] : in_flight_) {
        total += count;
    }
    return total;
}

void FifoScheduler::drain() { pending_.clear(); }

// ---------------------------------------------------------------------------
// ModelRouter (base.go baseRouter + group.go Group, merged; Option-B Engine lifecycle)
// ---------------------------------------------------------------------------

ModelRouter::ModelRouter(ServeOptions options, StartupObserver observer, BackendFactory factory)
    : base_options_(std::move(options)),
      observer_(std::move(observer)),
      swapper_(base_options_.model_config),
      scheduler_(1024, static_cast<int>(base_options_.max_concurrency)),
      factory_(std::move(factory)) {
    // The admission gate is the single concurrency setting, ServeOptions.max_concurrency (the
    // engine's compact-decode-batch count); it applies to every model's in-flight count.
    // No-resident, on-demand-load: the router launches UNLOADED. No backend is constructed at
    // startup -- the factory is NOT called, and resident_ / loaded_id_ stay at their empty
    // defaults. The first request for a model triggers the load path from the empty state
    // (do_swap from a null resident + the is_available readiness gate on /health); a loaded model
    // persists until the TTL ticker evicts it, then the next request reloads on demand. There is
    // no fixed "first model = resident/loaded" convention.
    // Start the 1s TTL ticker (process.go Start: a goroutine ticking every 1s, guarded by the
    // shutdown flag + condition variable). From the empty state the tick is a no-op (run_ttl_tick
    // returns early when loaded_id_ is empty); it only evicts a model the router last loaded.
    ttl_thread_ = std::thread([this] { ttl_loop(); });
}

ModelRouter::~ModelRouter() { shutdown(); }

ModelRouter::Grant
ModelRouter::route(std::string_view model_id_or_alias) {
    const ModelConfig* model = swapper_.resolve(model_id_or_alias);
    if (model == nullptr) {
        // Unknown id/alias. The HTTP handler maps this to a 404 (the next wiring step).
        throw std::out_of_range("model '" + std::string(model_id_or_alias) + "' not found");
    }
    const std::string target = model->id;

    std::unique_lock<std::mutex> lock(mu_);
    while (true) {
        if (!scheduler_.admit(target)) {
            // The unique_lock releases on throw (RAII). ConcurrencyLimit gate: reject.
            throw std::overflow_error("model '" + target + "': concurrency limit exceeded");
        }
        if (loaded_id_ == target && !swapping_) {
            // Fast path: the model is already the resident (no swap; the factory is NOT called).
            scheduler_.begin_in_flight(target);
            last_activity_ = std::chrono::steady_clock::now();
            lock.unlock();
            return Grant{resident_, [this, target] { release_in_flight(target); }};
        }
        if (swapping_) {
            // A swap is in flight (same or different target): join it (bounded queue + wait). The
            // joiner does NOT start a new swap; it queues in arrival order and is served once the
            // in-flight swap installs the model (or a different model installs, in which case the
            // loop starts a swap for this target).
            if (!scheduler_.enqueue(target)) {
                // Bounded queue full: reject (the unique_lock releases on throw).
                throw std::overflow_error("model '" + target + "': request queue full");
            }
            // Wait for the in-flight swap to finish (a swap installs the resident + notifies cv_).
            cv_.wait(lock, [this] { return !swapping_; });
            continue; // re-evaluate: fast path, or start a swap for a different target
        }
        // No swap in flight: start one (the starter does NOT enqueue itself; joiners do).
        swapping_ = true;
        swap_target_ = target;
        lock.unlock();
        do_swap(*model); // destroy resident + construct target + gate on is_available
        lock.lock();
        swapping_ = false;
        scheduler_.drain(); // queued requests are now served by the installed model
        cv_.notify_all();
        if (loaded_id_ != target) {
            // The swap failed to install (readiness timeout or shutdown). Reject; joiners that were
            // waiting now re-route (a fresh request will start a new swap).
            throw std::runtime_error("model '" + target + "': model is not ready");
        }
        continue; // re-evaluate: fast path grants this request
    }
}

std::shared_ptr<ModelBackend> ModelRouter::resident() const {
    std::lock_guard<std::mutex> lock(mu_);
    return resident_;
}

std::string ModelRouter::loaded_id() const {
    std::lock_guard<std::mutex> lock(mu_);
    return loaded_id_;
}

void ModelRouter::unload() {
    std::string evicted_id;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto evicted = std::move(resident_); // destroyed at scope exit (frees VRAM)
        if (evicted) { evicted_id = evicted->id(); }
        loaded_id_.clear();
    }
    cv_.notify_all(); // wake any joiners; they re-route
    if (!evicted_id.empty()) {
        // Explicit unload (design §5.2), fired OUTSIDE mu_ (non-blocking w.r.t. the router).
        emit_status(ModelStatusEvent{"unloaded", evicted_id, 0.0, 0});
    }
}

void ModelRouter::shutdown() {
    bool started;
    {
        std::lock_guard<std::mutex> lock(shutdown_mu_);
        started = !shutdown_;
        shutdown_ = true;
    }
    if (!started) {
        return; // idempotent (the destructor may call this after the user did)
    }
    {
        // Clear the status hook before the final unload: at shutdown there is no live /models/sse
        // subscriber (the server has stopped), and the router is destroyed AFTER the serving
        // HttpServer that owns the hook -- a late unload firing the hook would dereference a
        // destroyed server. In-service transitions still fire the hook (only this final
        // shutdown-path unload drops its event, which has no audience).
        std::lock_guard<std::mutex> lock(mu_);
        status_hook_ = StatusHook{};
    }
    cv_shutdown_.notify_all();
    if (ttl_thread_.joinable()) {
        ttl_thread_.join();
    }
    unload();
}

void ModelRouter::run_ttl_tick(std::chrono::steady_clock::time_point now) {
    std::string evicted_id;
    bool evicting = false;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (loaded_id_.empty() || swapping_) {
            return; // nothing loaded, or a swap is in flight (don't unload mid-swap)
        }
        const ModelConfig* resident = swapper_.resolve(loaded_id_);
        // effective_ttl = model.ttl if > 0 else global_ttl; <= 0 means no auto-unload.
        const int effective_ttl =
            (resident != nullptr && resident->ttl > 0) ? resident->ttl
                                                        : base_options_.model_config.global_ttl;
        if (effective_ttl <= 0) {
            return;
        }
        // An in-flight generation pins the backend: a model with live requests is never evicted,
        // even when it has been idle past its effective TTL (its Grant keeps the backend alive for
        // the request's lifetime; evicting here would strand the in-flight generation and let the
        // next request load a second Engine). Skip the unload this tick and revisit on a later tick.
        if (scheduler_.in_flight(loaded_id_) > 0) {
            return;
        }
        const int64_t idle_seconds =
            std::chrono::duration_cast<std::chrono::seconds>(now - last_activity_).count();
        if (idle_seconds > effective_ttl) {
            auto evicted = std::move(resident_); // destroyed at block exit (frees VRAM)
            if (evicted) { evicted_id = evicted->id(); }
            loaded_id_.clear();
            evicting = true;
            cv_.notify_all();
        }
    }
    if (evicting) {
        // TTL eviction (design §5.2): the state mutation ran under the lock; the hook fires after
        // it is released (OUTSIDE mu_, non-blocking w.r.t. the router).
        emit_status(ModelStatusEvent{"unloaded", evicted_id, 0.0, 0});
    }
}

int ModelRouter::swap_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return swap_count_;
}

bool ModelRouter::is_swapping() const {
    std::lock_guard<std::mutex> lock(mu_);
    return swapping_;
}

int ModelRouter::queued() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(scheduler_.queued());
}

int ModelRouter::in_flight() const {
    std::lock_guard<std::mutex> lock(mu_);
    return scheduler_.in_flight_total();
}

RouterStatusSnapshot ModelRouter::status_snapshot() const {
    std::lock_guard<std::mutex> lock(mu_);
    return RouterStatusSnapshot{loaded_id_, swapping_};
}

void ModelRouter::set_status_hook(StatusHook hook) {
    std::lock_guard<std::mutex> lock(mu_);
    status_hook_ = std::move(hook);
}

void ModelRouter::emit_status(const ModelStatusEvent& ev) {
    StatusHook hook;
    {
        std::lock_guard<std::mutex> lock(mu_); // copy the hook under the lock
        hook = status_hook_;
    }
    if (hook) {
        // Invoke OUTSIDE mu_ (the copy is held locally): a blocking subscriber cannot stall the
        // router. The hook is a thin, non-blocking serving callback (the hub's bounded-queue
        // fan-out, design §5.2).
        hook(ev);
    }
}

void ModelRouter::do_swap(const ModelConfig& target) {
    // 1. Destroy the current resident (frees VRAM BEFORE loading the new one). Option-B:
    //    process.go Stop on the evicted (the Engine has no unload(); destruction IS the unload).
    std::shared_ptr<ModelBackend> evicted;
    std::string evicted_id;
    {
        std::lock_guard<std::mutex> lock(mu_);
        evicted = std::move(resident_); // moved out (no longer the resident)
        if (evicted) { evicted_id = evicted->id(); }
        loaded_id_.clear();
    }
    evicted.reset(); // VRAM freed outside the lock

    // Status hook (design §5.2), fired OUTSIDE mu_ (do_swap runs with route()'s lock released):
    // a swap is a load (empty state) or a swap (a resident was evicted). If a model was evicted it
    // is now UNLOADED; the target is now LOADING. A blocking subscriber cannot stall the router
    // (the hook is copied under mu_ then invoked here, outside it).
    if (!evicted_id.empty()) {
        emit_status(ModelStatusEvent{"unloaded", evicted_id, 0.0, 0});
    }
    emit_status(ModelStatusEvent{"loading", target.id, 0.0, 0});

    // 2. Construct the target backend (Option-B: build the Engine). A load failure (a bad
    //    artifact, an OOM) throws from the factory: reset the swap state so the router is not
    //    stuck (swapping_ would stay true and every later route() would join the never-ending
    //    swap), then rethrow so the handler maps it to a 503.
    std::unique_ptr<ModelBackend> backend;
    try {
        backend = factory_(target);
    } catch (...) {
        std::lock_guard lock(mu_);
        swapping_ = false;
        swap_target_.clear();
        scheduler_.drain(); // queued joiners re-route (a fresh request retries the load)
        cv_.notify_all();
        throw;
    }

    // 3. Readiness gate (process.go healthCheck/EnsureReady): poll is_available under
    //    healthCheckTimeout. A real Engine is ready immediately after construction; the fake
    //    controls this to exercise the gate + the "loading" window. shutdown_ is read under its
    //    own guard (shutdown_mu_): the gate's 10 ms poll must not stall on route()'s mu_ while a
    //    request is routed (the lock pair mirrors shutdown() / ttl_loop()).
    const int hct = base_options_.model_config.health_check_timeout > 0
                        ? base_options_.model_config.health_check_timeout
                        : 120; // DefaultHealthCheckTimeoutSeconds (process.go)
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(hct);
    bool ready = false;
    while (true) {
        if (backend->is_available()) {
            ready = true;
            break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        {
            std::lock_guard<std::mutex> lock(shutdown_mu_);
            if (shutdown_) {
                break;
            }
        }
    }

    // 4. Install the new resident (only if ready and not shutting down; otherwise the backend is
    //    destroyed at scope exit and route() rejects the request). The install takes mu_ for the
    //    router state and shutdown_mu_ for the flag; the flag is copied under its own guard so the
    //    install decision and the state mutation are consistent with shutdown().
    bool installed = false;
    {
        bool shutting_down;
        {
            std::lock_guard<std::mutex> lock(shutdown_mu_);
            shutting_down = shutdown_;
        }
        std::lock_guard<std::mutex> lock(mu_);
        if (ready && !shutting_down) {
            resident_ = std::shared_ptr<ModelBackend>(std::move(backend));
            loaded_id_ = target.id;
            ++swap_count_;
            last_activity_ = std::chrono::steady_clock::now();
            installed = true;
        }
    }
    if (installed) {
        // swap-complete / load-complete (design §5.2): the target is now the resident, fired
        // OUTSIDE mu_ (non-blocking w.r.t. the router).
        emit_status(ModelStatusEvent{"loaded", target.id, 0.0, 0});
    }
}

void ModelRouter::ttl_loop() {
    while (true) {
        {
            std::unique_lock<std::mutex> lock(shutdown_mu_);
            cv_shutdown_.wait_for(lock, std::chrono::seconds(1),
                                  [this] { return shutdown_; });
            if (shutdown_) {
                return;
            }
        }
        run_ttl_tick(); // acquires mu_ briefly; a no-op when nothing is loaded or a swap is in flight
    }
}

void ModelRouter::release_in_flight(std::string_view model) {
    std::lock_guard<std::mutex> lock(mu_);
    scheduler_.release_in_flight(model);
}

} // namespace ninfer::serve
