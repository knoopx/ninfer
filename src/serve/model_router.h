#pragma once

// In-process multi-model ROUTER for ninfer-serve (STEP 2 of the multi-model port).
//
// This is the Option-B adaptation of llama-swap's router: instead of a child-process lifecycle
// (llama-swap Process interface: Run/EnsureReady/Stop), the router owns an in-process Engine
// lifecycle (construct + gate on is_available / destroy). Every other behavior is a direct port:
//
// NO-RESIDENT, ON-DEMAND-LOAD: the router launches with NO loaded backend (no Engine in VRAM,
// no startup warm-up). The first request for a model loads it on demand (a do_swap from the
// empty state + the is_available readiness gate on /health). A loaded model persists until the
// TTL ticker evicts it (idle > effective ttl); the next request reloads it on demand. There is
// no fixed "first model = resident/loaded" convention; the live loaded state is whatever the
// router last loaded (if not yet evicted).
//
//   - internal/router/group.go  groupSwapper.EvictionFor   -> GroupSwapper
//   - internal/router/base.go   baseRouter.doSwap          -> ModelRouter::do_swap
//   - internal/process/process.go Process state machine    -> ModelBackendState + TTL ticker
//   - internal/router/scheduler/fifo.go FIFO               -> FifoScheduler
//
// The router keeps ONE resident model (the single-exclusive-group case). A request for a
// non-resident model performs a swap: destroy the resident Engine (frees VRAM) + construct the
// target Engine + gate on is_available under healthCheckTimeout. The TTL ticker unloads the
// resident when it has been idle longer than its effective ttl.
//
// The HTTP handlers are NOT wired here (that is the next step); this module is self-contained,
// unit-tested with a fake backend, and compiles into ninfer_serve.

#include "serve/generation_service.h"
#include "serve/model_config.h"
#include "serve/serve_options.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ninfer::serve {

// llama-swap ProcessState (internal/process/process.go). The Option-B seam: the readiness of a
// backend is gated on is_available() (process.go healthCheck polls CheckEndpoint until 200); the
// terminal "shutdown" state is handled at the router (the TTL ticker / shutdown flag), not on the
// backend, so it is omitted from the per-backend vocabulary.
enum class ModelBackendState : std::uint8_t {
    Stopped,
    Starting,
    Ready,
    Stopping,
};

// The Option-B seam that replaces llama-swap's `Process` interface. The router depends only on
// this interface so the host-side unit test can inject a fake (no Engine, no GPU). The minimal
// virtual surface mirrors the GenerationService methods the router actually drives.
class ModelBackend {
public:
    virtual ~ModelBackend() = default;

    // The model id this backend serves (the public ModelConfig.id).
    [[nodiscard]] virtual std::string id() const = 0;

    // The /health readiness gate (process.go EnsureReady polls CheckEndpoint until 200).
    [[nodiscard]] virtual bool is_available() const = 0;

    // Generation surface the handler forwards to (the wiring step uses these on the granted
    // backend).
    [[nodiscard]] virtual PreparedRequest
    prepare(const GenerationRequest& req, GenerationConsumerMode consumer_mode,
            ninfer::GenerationObservationOptions observation,
            std::function<bool()> is_cancelled, ContextCacheHints context_cache) const = 0;
    [[nodiscard]] virtual int
    count_prompt_tokens(const GenerationRequest& req, std::function<bool()> is_cancelled) const = 0;
    virtual GenerationOutcome run(PreparedRequest& prepared, const StreamSink* sink,
                                  std::function<bool()> is_cancelled) = 0;

    // Diagnostics / warmup (trivially forwarded by the production backend).
    [[nodiscard]] virtual ninfer::LoadSummary load_summary() const = 0;
    virtual void warmup() = 0;

    // Serving diagnostics the HttpServer reads off the granted/resident backend (mirror the
    // GenerationService surface; the production backend forwards each to its inner service_).
    [[nodiscard]] virtual ninfer::RuntimeStats runtime_stats() const = 0;
    [[nodiscard]] virtual ninfer::MemorySummary memory_summary() const = 0;
    [[nodiscard]] virtual ninfer::ModelSamplingDefaults sampling_defaults() const = 0;
    [[nodiscard]] virtual const ninfer::EngineOptions& engine_options() const = 0;
};

// Production backend. Wraps a GenerationService (composition; GenerationService does NOT inherit
// ModelBackend). Built by copying the base ServeOptions and setting artifact_path = model.artifact
// (the per-model input for a swap); the GenerationService ctor then builds the Engine from it.
class EngineModelBackend final : public ModelBackend {
public:
    EngineModelBackend(const ModelConfig& model, ServeOptions base_options,
                       StartupObserver observer);

    [[nodiscard]] std::string id() const override;
    [[nodiscard]] bool is_available() const override;
    [[nodiscard]] PreparedRequest
    prepare(const GenerationRequest& req, GenerationConsumerMode consumer_mode,
            ninfer::GenerationObservationOptions observation, std::function<bool()> is_cancelled,
            ContextCacheHints context_cache) const override;
    [[nodiscard]] int
    count_prompt_tokens(const GenerationRequest& req, std::function<bool()> is_cancelled) const override;
    GenerationOutcome run(PreparedRequest& prepared, const StreamSink* sink,
                          std::function<bool()> is_cancelled) override;
    [[nodiscard]] ninfer::LoadSummary load_summary() const override;
    void warmup() override;
    [[nodiscard]] ninfer::RuntimeStats runtime_stats() const override;
    [[nodiscard]] ninfer::MemorySummary memory_summary() const override;
    [[nodiscard]] ninfer::ModelSamplingDefaults sampling_defaults() const override;
    [[nodiscard]] const ninfer::EngineOptions& engine_options() const override;

    // The merged per-model ServeOptions this backend was constructed with (base ServeOptions +
    // per-model overrides from the serve config). Serving diagnostics log the effective preset
    // from this value at model load.
    [[nodiscard]] const ServeOptions& serve_options() const;

private:
    std::string model_id_;
    std::unique_ptr<GenerationService> service_;
};

// llama-swap groupSwapper (internal/router/group.go) + FindConfig (config.go). Builds the
// id->model, alias->id, and model->group maps from the Config. A model not named in any group is
// its own implicit exclusive group (the single-resident semantics this router targets).
class GroupSwapper {
public:
    explicit GroupSwapper(const Config& config);

    // Resolve a public id or any alias to its ModelConfig; nullptr if unknown (the handler maps
    // this to a 404).
    [[nodiscard]] const ModelConfig* resolve(std::string_view id_or_alias) const;

    // The group id a model belongs to (its own id when it is not a member of any group).
    [[nodiscard]] std::string group_of(std::string_view model) const;

    // ground: group.go groupSwapper.EvictionFor. Given `target` (the model being loaded) and the
    // `running` set, returns the set to evict. Same-group + swap => evict the siblings;
    // cross-group + exclusive => evict unless the other group is persistent. In the
    // single-exclusive-group (one resident) case this reduces to: if target != resident, evict the
    // resident.
    [[nodiscard]] std::vector<std::string>
    eviction_for(const std::string& target, const std::vector<std::string>& running) const;

private:
    [[nodiscard]] const GroupConfig* group(std::string_view group_id) const;

    Config config_;
    std::unordered_map<std::string, const ModelConfig*> model_by_id_;
    std::unordered_map<std::string, std::string> alias_to_id_;
    std::unordered_map<std::string, std::string> model_to_group_;
};

// llama-swap FIFO scheduler (internal/router/scheduler/fifo.go). Bounded FIFO ingress: requests are
// held in arrival order and drained front-to-back when a swap completes (the queue is bounded;
// drop/reject when full). Unlike the llama-swap sketch's "A B C A B C -> A A B B C C" cross-model
// reordering, this router has no cross-model ordering guarantee: route() drains the queue after
// each swap (drain()) and joiners re-route in wake order, so only bounded, arrival-ordered ingress
// with drop/reject when full holds.
//
// Signatures are the decision-tree helpers that mirror fifo.go so the router reads faithfully. Two
// deliberate, documented deviations from the sketch:
//   - enqueue() returns bool (true=accepted, false=dropped-when-full) so the bounded rejection is
//     observable (the sketch says void but "drop/reject when full" must be reportable).
//   - pop_front() is non-const and returns std::string (owned): a const pop would need a mutable
//     queue, and a returned string_view would dangle once the front element is removed.
class FifoScheduler {
public:
    // max_queue_size bounds the pending queue (drop/reject when full). Default 1024. limit is
    // ServeOptions.max_concurrency (the engine's compact-decode-batch count): the single
    // concurrency setting, and the router's admission gate for every model.
    explicit FifoScheduler(std::size_t max_queue_size = 1024, int limit = 1);

    // Concurrency gate (fifo.go admit): may this model take one more in-flight request?
    // in_flight(model) < limit.
    [[nodiscard]] bool admit(std::string_view model) const;

    // Bounded enqueue (fifo.go enqueue): append to the arrival-ordered pending queue; returns
    // false (drop) when the queue is at max_queue_size.
    [[nodiscard]] bool enqueue(std::string_view model);

    // Remove and return the front pending model (arrival order); nullopt when empty.
    [[nodiscard]] std::optional<std::string> pop_front();

    // 1-based position of the FIRST pending request for `model`; 0 when none is queued.
    // Ground: fifo.go broadcastQueuePositions.
    [[nodiscard]] int queue_position(std::string_view model) const;

    // Number of pending requests held (the router's `queued()` introspection hook).
    [[nodiscard]] std::size_t queued() const;

    // In-flight accounting: begin_in_flight reserves a slot when a request is granted; the Grant
    // releases it on destruction. Ground: fifo.go inFlight++ / OnServeDone inFlight--.
    void begin_in_flight(std::string_view model);
    void release_in_flight(std::string_view model);
    [[nodiscard]] int in_flight(std::string_view model) const;
    [[nodiscard]] int in_flight_total() const;

    // Clear the pending queue (the queued requests are now served by the drained swap).
    void drain();

private:
    std::deque<std::string> pending_;
    std::unordered_map<std::string, int> in_flight_;
    std::size_t max_queue_size_;
    int limit_;
};

// llama-swap baseRouter (base.go) + Group (group.go), merged for the one-loaded-model case.
// Holds AT MOST ONE loaded model at a time (the single-exclusive-group / one-GPU case) and
// STARTS WITH NONE (no-resident, on-demand-load). Owns the Engine lifecycle (construct / gate
// on is_available / destroy), the selection/eviction policy (GroupSwapper), the bounded FIFO
// ingress (FifoScheduler), and the 1s TTL ticker thread (which evicts the last-loaded model
// when idle; from the empty state the tick is a no-op).
// A model-status change pushed to GET /models/sse subscribers (llama-parity real-design §5.2).
// `status` is the webui's Yo wire vocabulary (unloaded|loading|loaded|sleeping|failed) -- the same
// vocabulary the minimal ModelSseHub record already uses (design §3.3); `progress` is reported
// only while loading (the webui reads it then); `exit_code` is 0 on success, non-zero on a failed
// transition. The router's status hook (ModelRouter::set_status_hook) fires one of these at each
// status transition, OUTSIDE the router lock (the router never invokes the hook while holding
// mu_), so a subscriber that blocks cannot stall route(). Serving-owned: HttpServer::attach sets
// the hook to the ModelSseHub's fan-out; a null hook (the default) is a no-op, so the router
// still works standalone (no /models/sse subscriber attached).
struct ModelStatusEvent {
    std::string status;  // "unloaded" | "loading" | "loaded" | "sleeping" | "failed"
    std::string model;   // the model id (the target on a load/swap; the evicted id on an unload)
    double progress      = 0.0;  // 0..1; meaningful only while status == "loading"
    int exit_code        = 0;    // 0 on success; non-zero on a failed transition
};

// A consistent snapshot of the router's live model state, taken under a single lock so a status
// record never pairs a "loading" marker with a stale model id (a swap may complete between two
// separate is_swapping()/loaded_id() reads). Consumers that render a live status (e.g. the
// /models/sse connect record) use this instead of the individual introspection hooks.
struct RouterStatusSnapshot {
    std::string loaded_id;  // the loaded model id ("" when nothing is loaded or mid-swap)
    bool swapping = false;  // a load/swap is in flight
};

class ModelRouter {
public:
    using BackendFactory = std::function<std::unique_ptr<ModelBackend>(const ModelConfig&)>;
    // The status hook invoked at each model-status transition (load/swap-start/swap-complete/
    // unload/TTL-evict). Set by the serving layer (HttpServer::attach wires it to the
    // ModelSseHub fan-out). The router invokes it OUTSIDE mu_ (never while holding the router
    // lock), so a blocking subscriber cannot stall route().
    using StatusHook = std::function<void(const ModelStatusEvent&)>;

    // The RAII grant: keeps the granted backend alive for the request (shared_ptr) and, on
    // destruction, releases the request's in-flight reservation so the TTL ticker only unloads a
    // model that is idle AND has no in-flight requests (an in-flight generation pins the backend).
    // The release callback captures the owning router; a Grant must not outlive the router that
    // issued it (the operator shuts the router down before it is destroyed, so this holds in
    // practice).
    struct Grant {
        std::shared_ptr<ModelBackend> backend; // keeps the model alive for the request
        std::function<void()> release_in_flight; // decrements the in-flight count on destruction
        Grant() = default;
        Grant(std::shared_ptr<ModelBackend> b, std::function<void()> r)
            : backend(std::move(b)), release_in_flight(std::move(r)) {}
        // A Grant is a single reservation: moving transfers it; copying would double-release.
        Grant(const Grant&)            = delete;
        Grant& operator=(const Grant&) = delete;
        Grant(Grant&&) noexcept        = default;
        Grant& operator=(Grant&&) noexcept = default;
        ~Grant() {
            if (release_in_flight) { release_in_flight(); }
        }
        // Release the reservation (decrement the in-flight count) and drop the backend reference
        // (the VRAM is freed when no other grant references the backend). Idempotent.
        void reset() {
            if (release_in_flight) {
                release_in_flight();
                release_in_flight = nullptr;
            }
            backend.reset();
        }
    };

    // Constructs the router: copies the per-model concurrency limits into the scheduler and starts
    // the 1s TTL ticker thread. The router starts with NO loaded backend (no-resident,
    // on-demand-load); the factory is NOT called at construction -- the first request for a model
    // triggers the load path from the empty state (do_swap from a null resident + the readiness
    // gate). The factory must not throw when it is later called for a load; a load failure
    // surfaces via route()'s "model is not ready" rejection.
    ModelRouter(ServeOptions options, StartupObserver observer, BackendFactory factory);

    ~ModelRouter();
    ModelRouter(const ModelRouter&)            = delete;
    ModelRouter& operator=(const ModelRouter&) = delete;
    ModelRouter(ModelRouter&&)                 = delete;
    ModelRouter& operator=(ModelRouter&&)      = delete;

    // Resolve id/alias -> grant a backend, performing a swap (destroy resident + construct target +
    // gate on is_available under healthCheckTimeout) when the model is not the resident. This is
    // the Option-B doSwap. Unknown id/alias throws std::out_of_range ("model '<id>' not found"; the
    // handler maps it to 404). A concurrency-limit breach throws std::overflow_error (the handler
    // maps it to a 429/503-style rejection). The returned Grant keeps the backend alive for the
    // request and releases the in-flight reservation on destruction.
    Grant route(std::string_view model_id_or_alias);

    // Set (or clear, by passing an empty hook) the status hook fired at each status transition
    // (load/swap-start/swap-complete/unload/TTL-evict), OUTSIDE the router lock. A null hook is a
    // no-op (the router works standalone with no /models/sse subscriber). Called once by
    // HttpServer::attach; the serving layer owns the hook (the router only stores + fires it).
    void set_status_hook(StatusHook hook);

    // The currently-loaded backend (a copy of the shared_ptr, keeping it alive); null when
    // nothing is loaded (at startup, mid-swap, or after a TTL eviction). The "resident" name
    // is retained for the loaded-model fast path; it is NOT a pre-loaded model.
    [[nodiscard]] std::shared_ptr<ModelBackend> resident() const;
    // The id of the loaded model; "" when nothing is loaded (always "" at startup).
    [[nodiscard]] std::string loaded_id() const;
    // Destroy the resident (TTL / eviction) so the next request reloads it.
    void unload();
    // Stop the TTL ticker thread, then unload the resident. Idempotent.
    void shutdown();
    // One TTL tick: if the loaded model has been idle longer than its effective ttl (model.ttl if
    // > 0 else global_ttl; <= 0 means no auto-unload) AND has no in-flight requests (an in-flight
    // generation pins the backend, so it is never evicted mid-request), unload it. Skipped
    // entirely while a swap is in flight. Callable directly with a synthetic `now` so the host-side
    // test needs no sleep.
    void run_ttl_tick(std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());

    // Introspection hooks for the unit test (idempotent state reads).
    [[nodiscard]] int swap_count() const; // completed swaps (destroy + construct)
    [[nodiscard]] bool is_swapping() const; // a swap is in flight
    [[nodiscard]] int queued() const; // pending requests held by the FIFO scheduler
    [[nodiscard]] int in_flight() const; // granted-but-not-yet-completed requests
    // A single-lock consistent snapshot of loaded_id + swapping (see RouterStatusSnapshot); the
    // /models/sse connect record uses this so its status and model id cannot diverge.
    [[nodiscard]] RouterStatusSnapshot status_snapshot() const;

private:
    // Option-B doSwap (base.go doSwap): destroy the current resident (frees VRAM; the Engine has
    // no unload()), construct the target via the factory, then gate on readiness: loop
    // `while (!ready && elapsed < health_check_timeout)` (poll is_available; a real Engine is ready
    // immediately after construction, the fake controls this to exercise the gate + the "loading"
    // window). Sets resident_ + loaded_id_ and increments swap_count_. Called OUTSIDE the router
    // lock by route() so is_swapping() is observable to concurrent requests.
    void do_swap(const ModelConfig& target);
    // Fire the status hook (if set) with `ev`: copy the hook under mu_, then invoke it OUTSIDE
    // mu_, so a subscriber that blocks cannot stall the router (the hook is a thin, non-blocking
    // serving callback -- the hub's bounded-queue fan-out, design §5.2).
    void emit_status(const ModelStatusEvent& ev);
    // The 1s ticker loop body (guarded by the shutdown flag + shutdown condition variable).
    void ttl_loop();
    // Release a granted request's in-flight reservation (called by Grant::~Grant).
    void release_in_flight(std::string_view model);

    // --- state (guarded by mu_) ---
    // The /models/sse fan-out hook (set by the serving layer; fired OUTSIDE mu_). Cleared in
    // shutdown() so the final unload at shutdown cannot invoke a stale/dangling hook (the router is
    // destroyed AFTER the serving HttpServer that owns the hook; a late unload would otherwise
    // dereference a destroyed server).
    StatusHook status_hook_;
    ServeOptions base_options_;
    StartupObserver observer_;
    GroupSwapper swapper_;
    FifoScheduler scheduler_;
    BackendFactory factory_;

    std::shared_ptr<ModelBackend> resident_;
    std::string loaded_id_;
    bool swapping_   = false;
    std::string swap_target_;
    int swap_count_  = 0;
    std::chrono::steady_clock::time_point last_activity_ = std::chrono::steady_clock::now();

    // --- synchronization ---
    // mutable so the const introspection hooks (resident/loaded_id/swap_count/queued/in_flight)
    // can lock it to read guarded state without copying.
    mutable std::mutex mu_; // guards all router state (resident_, loaded_id_, swapping_, swap_count_, ...)
    std::condition_variable cv_; // signals swap completion (joiners) / unload
    std::mutex shutdown_mu_; // guards shutdown_ (separate from mu_ so the ticker's 1s wait never
    std::condition_variable cv_shutdown_; // stalls route()); cv_ uses mu_
    bool shutdown_ = false;
    std::thread ttl_thread_;
};

} // namespace ninfer::serve
