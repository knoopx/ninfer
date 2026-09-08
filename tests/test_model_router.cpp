// Unit test for the in-process multi-model router (src/serve/model_router.{h,cpp}).
//
// A FakeModelBackend (no Engine, no GPU) is injected via the router's BackendFactory so the host
// test can drive construction/destruction counts, the readiness gate, and a construction block
// (simulating a slow load for the concurrent-queueing test). The assertions map to the delegation
// acceptance matrix:
//
// NO-RESIDENT, ON-DEMAND-LOAD router (the corrected semantics): the router launches with nothing
// loaded. The assertions map to the acceptance matrix:
//
//   1. On-demand first load  - the router starts empty; the first route() loads the model from the
//      empty state (one load/swap), and a second route() of the same model is the loaded-model
//      fast path (no second load).
//   2. Load from empty + swap back - route(other) loads other from the empty state; route(res)
//      swaps other -> res (the previously-loaded model is destroyed on the swap).
//   3. Concurrent queueing   - while a load/swap is in flight, a second request queues and is
//      served after the swap installs the model.
//   4. FIFO ordering         - queued requests are served in arrival order (drain front-to-back).
//   5. TTL idle unload       - run_ttl_tick(now_past_ttl) unloads an idle model; the next request
//      reloads it.
//   6. Concurrency limit     - the (N+1)-th in-flight request is rejected.
//   7. No-resident lifecycle - start unloaded (0 constructions, loaded_id empty) -> first request
//      loads on demand -> serves (fast path, no re-load) -> TTL evicts -> next request reloads.
//
// No gtest: a bare C++ main() with a small CHECK macro; the binary exits non-zero on any failure.

#include "serve/model_router.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ns = ninfer::serve;
using ns::FifoScheduler;
using ns::ModelBackend;
using ns::ModelConfig;
using ns::ModelRouter;
using ns::Config;
using ns::GroupSwapper;

// ---------------------------------------------------------------------------
// Test-controlled backend state + the fake backend
// ---------------------------------------------------------------------------

struct FakeState {
    // Construction/destruction accounting (guarded by mu).
    mutable std::mutex mu;
    std::unordered_map<std::string, int> constructed;
    std::unordered_map<std::string, int> destroyed;

    // The "loading" gate: when closed, a non-resident fake blocks in its constructor (a slow load).
    std::atomic<bool> gate_closed{false};
    std::mutex gate_mu;
    std::condition_variable gate_cv;
    // A fresh non-resident construction in progress (so the test can wait for A to enter the swap).
    std::atomic<int> in_construction{0};
    std::mutex construct_mu;
    std::condition_variable construct_cv;

    int constructed_of(std::string_view id) const {
        std::lock_guard lock(mu);
        const auto it = constructed.find(std::string(id));
        return it != constructed.end() ? it->second : 0;
    }
    int destroyed_of(std::string_view id) const {
        std::lock_guard lock(mu);
        const auto it = destroyed.find(std::string(id));
        return it != destroyed.end() ? it->second : 0;
    }
    int constructions_total() const {
        std::lock_guard lock(mu);
        int total = 0;
        for (const auto& [_, count] : constructed) {
            total += count;
        }
        return total;
    }
};

class FakeModelBackend final : public ModelBackend {
public:
    FakeModelBackend(std::string id, FakeState* state, bool blocks_when_gate_closed)
        : id_(std::move(id)), state_(state), blocks_(blocks_when_gate_closed) {
        {
            std::lock_guard lock(state_->mu);
            state_->constructed[id_]++;
        }
        if (blocks_) {
            // Simulate a slow load: block in construction until the gate opens.
            ++state_->in_construction;
            state_->construct_cv.notify_all();
            {
                std::unique_lock<std::mutex> lock(state_->gate_mu);
                state_->gate_cv.wait(lock,
                                     [this] { return !this->state_->gate_closed.load(); });
            }
            --state_->in_construction;
        }
    }
    ~FakeModelBackend() override {
        std::lock_guard lock(state_->mu);
        state_->destroyed[id_]++;
    }

    std::string id() const override { return id_; }
    // Ready once constructed (the fake has no async readiness; the construction block is the
    // "loading" window the concurrent test exercises).
    bool is_available() const override { return true; }

    ns::PreparedRequest
    prepare(const ns::GenerationRequest&, ns::GenerationConsumerMode,
            ninfer::GenerationObservationOptions, std::function<bool()>,
            ninfer::ContextCacheHints) const override {
        return {};
    }
    int count_prompt_tokens(const ns::GenerationRequest&, std::function<bool()>) const override {
        return 0;
    }
    ns::GenerationOutcome run(ns::PreparedRequest&, const ns::StreamSink*,
                              std::function<bool()>) override {
        return {};
    }
    ninfer::LoadSummary load_summary() const override { return {}; }
    void warmup() override {}
    // The 4 diagnostic passthroughs the HttpServer reads; the fake returns default-constructed
    // values (no Engine) so the router module stays host-testable.
    ninfer::RuntimeStats runtime_stats() const override { return {}; }
    ninfer::MemorySummary memory_summary() const override { return {}; }
    ninfer::ModelSamplingDefaults sampling_defaults() const override { return {}; }
    const ninfer::EngineOptions& engine_options() const override {
        static const ninfer::EngineOptions engine; // a stable dummy (the fake has no Engine)
        return engine;
    }

private:
    std::string id_;
    FakeState* state_;
    bool blocks_;
};

// A backend whose readiness gate never succeeds (is_available always false) so the router's
// readiness gate times out; used to exercise the route() "model is not ready" rejection (the
// exception the F1 handlers now map to 503).
class NeverReadyBackend final : public ModelBackend {
public:
    explicit NeverReadyBackend(std::string id) : id_(std::move(id)) {}
    std::string id() const override { return id_; }
    bool is_available() const override { return false; }
    ns::PreparedRequest
    prepare(const ns::GenerationRequest&, ns::GenerationConsumerMode,
            ninfer::GenerationObservationOptions, std::function<bool()>,
            ninfer::ContextCacheHints) const override {
        return {};
    }
    int count_prompt_tokens(const ns::GenerationRequest&, std::function<bool()>) const override {
        return 0;
    }
    ns::GenerationOutcome run(ns::PreparedRequest&, const ns::StreamSink*,
                              std::function<bool()>) override {
        return {};
    }
    ninfer::LoadSummary load_summary() const override { return {}; }
    void warmup() override {}
    ninfer::RuntimeStats runtime_stats() const override { return {}; }
    ninfer::MemorySummary memory_summary() const override { return {}; }
    ninfer::ModelSamplingDefaults sampling_defaults() const override { return {}; }
    const ninfer::EngineOptions& engine_options() const override {
        static const ninfer::EngineOptions engine;
        return engine;
    }

private:
    std::string id_;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a two-model Config: "res" (the resident, first) and "other". Per-model ttl is
// parameterized so each test case controls its own policy (ttl=0 disables auto-unload, so the
// 1s TTL ticker thread is a no-op for the non-TTL cases and cannot interfere).
Config make_config(int other_ttl) {
    Config config;

    ModelConfig res;
    res.id = "res";
    res.aliases = {"res-alias"};
    res.artifact = "/tmp/res.ninfer";
    res.ttl = 0; // no auto-unload for the resident (the TTL case uses "other")
    config.models.push_back(res);

    ModelConfig other;
    other.id = "other";
    other.artifact = "/tmp/other.ninfer";
    other.ttl = other_ttl;
    config.models.push_back(other);

    config.global_ttl = 0; // per-model ttl governs; no global auto-unload
    config.health_check_timeout = 5;
    return config;
}

ModelRouter::BackendFactory make_factory(FakeState* state) {
    // The factory inspects the shared gate at call time: the resident (built at construction) is
    // built with the gate open; a non-resident swap sees the gate as the test set it.
    return [state](const ModelConfig& model) -> std::unique_ptr<ModelBackend> {
        const bool blocks = state->gate_closed.load();
        return std::make_unique<FakeModelBackend>(model.id, state, blocks);
    };
}

// A minimal ServeOptions with a default artifact path (the router's construction only needs a
// well-formed ServeOptions; the fakes never use it, but the type must be valid).
ns::ServeOptions make_options(const Config& config, int max_concurrency = 10) {
    ns::ServeOptions options;
    options.model_config = config;
    // artifact_path is a required non-empty string; the router copies it into the per-model
    // backend options (unused by the fakes, but it must be valid for the ServeOptions contract).
    options.artifact_path = "/tmp/default.ninfer";
    // The tests exercise the router's admission gate (the single concurrency setting,
    // ServeOptions.max_concurrency). The default 10 lets tests hold multiple concurrent grants
    // per model; test_6 lowers it to its gate under test.
    options.max_concurrency = static_cast<std::uint32_t>(max_concurrency);
    return options;
}

namespace {
int g_failures = 0;
}

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (cond) {                                                            \
            std::printf("  PASS: %s\n", msg);                                 \
        } else {                                                               \
            std::printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);     \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

// Wait (poll with a timeout) until `pred` holds; returns true if it did.
template <typename Pred>
bool wait_for(std::chrono::milliseconds timeout, Pred pred) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!pred()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

void test_1_on_demand_first_load() {
    std::printf("[test] 1. no-resident: start empty, first request loads on demand, second is the "
                "loaded-model fast path\n");
    FakeState state;
    Config config = make_config(/*other_ttl*/ 0);
    ModelRouter router(make_options(config), {}, make_factory(&state));

    // No-resident: nothing is loaded at startup; the factory was never called.
    CHECK(router.loaded_id().empty(), "no model is loaded at startup");
    CHECK(state.constructions_total() == 0, "the factory was not called at construction (no pre-load)");

    // The first request for a model loads it on demand (a load/swap from the empty state).
    ModelRouter::Grant grant = router.route("res");
    CHECK(router.swap_count() == 1, "the first request performs a load (a swap from the empty state)");
    CHECK(router.loaded_id() == "res", "the requested model is now loaded");
    CHECK(grant.backend != nullptr, "grant holds the loaded backend");
    CHECK(grant.backend->id() == "res", "granted backend is the requested model");
    CHECK(state.constructions_total() == 1, "the factory was called once to load the model on demand");
    CHECK(state.constructed_of("res") == 1, "the requested model was constructed");
    CHECK(state.constructed_of("other") == 0, "the other model was not constructed");
    CHECK(router.in_flight() == 1, "the granted request is in-flight");

    // A second request for the now-loaded model takes the loaded-model fast path: NO second load.
    ModelRouter::Grant fast = router.route("res");
    CHECK(router.swap_count() == 1, "the loaded-model fast path did not trigger another load");
    CHECK(state.constructions_total() == 1, "no second construction (the fast path reuses the loaded model)");
    CHECK(fast.backend != nullptr && fast.backend->id() == "res", "the fast path grants the loaded model");
}

void test_2_load_from_empty_and_swap_back() {
    std::printf("[test] 2. load from the empty state, then swap back\n");
    FakeState state;
    Config config = make_config(/*other_ttl*/ 0);
    ModelRouter router(make_options(config), {}, make_factory(&state));

    CHECK(router.loaded_id().empty(), "nothing is loaded at construction (no-resident)");
    ModelRouter::Grant grant = router.route("other"); // load "other" from the empty state
    CHECK(router.swap_count() == 1, "one load performed");
    CHECK(router.loaded_id() == "other", "the requested model is now loaded");
    CHECK(grant.backend != nullptr, "grant holds the loaded backend");
    CHECK(grant.backend->id() == "other", "granted backend is the loaded model");
    CHECK(state.constructed_of("other") == 1, "the requested model was constructed by the load");
    CHECK(state.constructed_of("res") == 0, "the other model was not constructed");
    CHECK(state.destroyed_of("res") == 0, "nothing was destroyed (the start state was empty, no pre-loaded resident)");

    // Swap the loaded model out for "res": the previously-loaded "other" is destroyed.
    ModelRouter::Grant back = router.route("res");
    CHECK(router.swap_count() == 2, "a second load (swap to the other model)");
    CHECK(router.loaded_id() == "res", "the other model is now loaded");
    CHECK(state.constructed_of("res") == 1, "the other model was constructed");
    // The swapped-out "other" backend is still referenced by the first grant (its shared_ptr);
    // only when no grant references it is it destroyed (VRAM freed).
    grant.reset();
    CHECK(state.destroyed_of("other") == 1, "releasing the grant freed the swapped-out model");
    (void)back;
}

void test_3_concurrent_queueing() {
    std::printf("[test] 3. concurrent request queues while a load/swap is in flight\n");
    FakeState state;
    Config config = make_config(/*other_ttl*/ 0);
    // The router starts unloaded (no-resident). The first request (the load) is what the gate
    // delays: close the gate so the "other" construction blocks (a slow load window).
    ModelRouter router(make_options(config), {}, make_factory(&state));
    CHECK(router.loaded_id().empty(), "nothing is loaded at construction (no-resident)");
    // Close the gate so the "other" construction blocks (a slow load window).
    state.gate_closed.store(true);

    ModelRouter::Grant a_grant, b_grant;
    bool b_done = false;

    std::thread A([&] { a_grant = router.route("other"); }); // the starter: enters do_swap, blocks
    CHECK(wait_for(std::chrono::milliseconds(500), [&] { return state.in_construction.load() > 0; }),
          "A entered the swap (a construction is in progress)");
    CHECK(router.is_swapping(), "the router reports a swap in flight");

    std::thread B([&] { b_grant = router.route("other"); b_done = true; }); // the joiner: queues
    CHECK(wait_for(std::chrono::milliseconds(500), [&] { return router.queued() == 1; }),
          "B's request is held in the FIFO queue while the swap is in flight");
    CHECK(!b_done, "the queued request has not completed while the swap is in flight");

    // Open the gate: the construction completes, the swap installs, and both requests are served.
    {
        std::lock_guard lock(state.gate_mu);
        state.gate_closed.store(false);
    }
    state.gate_cv.notify_all();
    A.join();
    B.join();

    CHECK(b_done, "the queued request completed once the swap finished");
    CHECK(router.loaded_id() == "other", "the swapped-in model is loaded");
    CHECK(router.swap_count() == 1, "exactly one swap (the joiner did not start a second)");
    CHECK(router.queued() == 0, "the FIFO queue was drained on swap completion");
    CHECK(a_grant.backend != nullptr && a_grant.backend->id() == "other",
          "the starter's grant is the swapped-in model");
    CHECK(b_grant.backend != nullptr && b_grant.backend->id() == "other",
          "the joiner's grant is the swapped-in model");
}

void test_4_fifo_ordering() {
    std::printf("[test] 4. FIFO ordering (arrival order, drain front-to-back) + bounded + gate\n");
    // The router's queue is the FifoScheduler; verify the "A B C A B C -> A A B B C C" ordering
    // property (arrival-order drain) directly on the scheduler.
    FifoScheduler scheduler(16);
    CHECK(scheduler.enqueue("a"), "enqueue a");
    CHECK(scheduler.enqueue("b"), "enqueue b");
    CHECK(scheduler.enqueue("c"), "enqueue c");
    CHECK(scheduler.enqueue("a"), "enqueue a (second)");
    CHECK(scheduler.enqueue("b"), "enqueue b (second)");
    CHECK(scheduler.enqueue("c"), "enqueue c (second)");
    CHECK(scheduler.queued() == 6, "six requests held in arrival order");
    CHECK(scheduler.queue_position("a") == 1, "first 'a' is at position 1");
    CHECK(scheduler.queue_position("b") == 2, "first 'b' is at position 2");
    CHECK(scheduler.pop_front() == "a", "drain front-to-back: a");
    CHECK(scheduler.pop_front() == "b", "drain front-to-back: b");
    CHECK(scheduler.pop_front() == "c", "drain front-to-back: c");
    CHECK(scheduler.pop_front() == "a", "drain front-to-back: a (second)");
    CHECK(scheduler.pop_front() == "b", "drain front-to-back: b (second)");
    CHECK(scheduler.pop_front() == "c", "drain front-to-back: c (second)");
    CHECK(scheduler.pop_front() == std::nullopt, "drained: no more pending");

    // Bounded: drop when full.
    FifoScheduler bounded(2);
    CHECK(bounded.enqueue("x"), "bounded enqueue x");
    CHECK(bounded.enqueue("y"), "bounded enqueue y");
    CHECK(!bounded.enqueue("z"), "bounded queue full: z is dropped");
    CHECK(bounded.queued() == 2, "bounded queue holds exactly its limit");

    // Concurrency gate: in_flight < limit (the single concurrency setting; here limit 2).
    FifoScheduler gated(16, 2);
    CHECK(gated.admit("m"), "admit when in_flight (0) < limit (2)");
    gated.begin_in_flight("m");
    CHECK(gated.admit("m"), "admit when in_flight (1) < limit (2)");
    gated.begin_in_flight("m");
    CHECK(!gated.admit("m"), "reject when in_flight (2) == limit (2)");
    CHECK(gated.in_flight("m") == 2, "in-flight count is 2");
    gated.release_in_flight("m");
    CHECK(gated.in_flight("m") == 1, "release decrements the in-flight count");
}

void test_5_ttl_idle_unload() {
    std::printf("[test] 5. TTL idle unload (run_ttl_tick(now_past_ttl) then reload)\n");
    FakeState state;
    // "other" has a small ttl (2s) so the effective ttl is small; the resident (ttl 0) and global
    // (0) disable auto-unload for everything else, so the 1s ticker is a no-op for this case.
    Config config = make_config(/*other_ttl*/ 2);
    ModelRouter router(make_options(config), {}, make_factory(&state));

    ModelRouter::Grant grant = router.route("other"); // load "other" (swap), last_activity_ = now
    CHECK(router.loaded_id() == "other", "the non-resident is loaded");
    // Release the grant so no in-flight request holds the model; the tick can then unload it and
    // the backend's VRAM is actually freed (the grant's shared_ptr would otherwise keep it alive).
    grant.reset();

    // The model is idle longer than its effective ttl (2s): unload it.
    router.run_ttl_tick(std::chrono::steady_clock::now() + std::chrono::seconds(5));
    CHECK(router.loaded_id().empty(), "the idle model was unloaded by the TTL tick");
    CHECK(state.destroyed_of("other") == 1, "the unloaded model's backend was destroyed");

    // The next request reloads it (a fresh swap).
    ModelRouter::Grant reload = router.route("other");
    CHECK(router.loaded_id() == "other", "the model was reloaded on the next request");
    CHECK(router.swap_count() == 2, "the reload was a swap (incremented the swap count)");
    CHECK(reload.backend != nullptr, "the reloaded grant is valid");
    (void)reload;
}

void test_6_max_concurrency_gate() {
    std::printf("[test] 6. concurrency limit: the (N+1)-th in-flight request is rejected\n");
    FakeState state;
    const int limit = 2; // the single concurrency setting (ServeOptions.max_concurrency)
    Config config = make_config(/*other_ttl*/ 0);
    ModelRouter router(make_options(config, limit), {}, make_factory(&state));

    // Hold `limit` in-flight grants for "other" (the swap loads it; the grants reserve slots).
    std::vector<ModelRouter::Grant> held;
    held.push_back(router.route("other")); // swap + grant (in-flight = 1)
    CHECK(router.in_flight() == 1, "one in-flight after the first grant");
    held.push_back(router.route("other")); // fast path + grant (in-flight = 2)
    CHECK(router.in_flight() == limit, "in-flight reached the limit");

    // The (N+1)-th in-flight request must be rejected (the concurrency gate).
    bool rejected = false;
    try {
        router.route("other"); // in-flight (2) == limit (2) -> reject
    } catch (const std::overflow_error&) {
        rejected = true;
    }
    CHECK(rejected, "the (N+1)-th in-flight request is rejected (overflow_error)");

    // Release one grant: the (N+1)-th is now admitted again.
    held.pop_back(); // releases in-flight 2 -> 1
    ModelRouter::Grant again = router.route("other"); // in-flight 1 < 2 -> admit
    CHECK(router.in_flight() == 2, "after releasing a grant, the next request is admitted");
    CHECK(again.backend != nullptr, "the admitted request holds a valid grant");
}

void test_7_no_resident_lifecycle() {
    std::printf("[test] 7. no-resident lifecycle: start unloaded -> load on demand -> serve -> "
                "TTL evict -> reload\n");
    FakeState state;
    // A single model with a short ttl (2s) so the TTL tick can evict it; global_ttl = 0 so no
    // other policy interferes.
    Config config;
    ModelConfig m;
    m.id = "m";
    m.artifact = "/tmp/m.ninfer";
    m.ttl = 2; // effective ttl 2s (auto-unload after idle > 2s)
    config.models.push_back(m);
    config.global_ttl = 0;
    config.health_check_timeout = 5;

    ModelRouter router(make_options(config), {}, make_factory(&state));

    // 1. Start UNLOADED: nothing constructed, nothing loaded.
    CHECK(router.loaded_id().empty(), "starts with no model loaded (no-resident)");
    CHECK(state.constructions_total() == 0, "nothing is pre-loaded (the factory was never called)");

    // 2. The first request loads the model ON DEMAND and serves it.
    ModelRouter::Grant serve = router.route("m");
    CHECK(router.swap_count() == 1, "the first request loaded the model on demand");
    CHECK(router.loaded_id() == "m", "the model is loaded after the first request");
    CHECK(serve.backend != nullptr && serve.backend->id() == "m",
          "the first request is served by the loaded model");
    CHECK(state.constructions_total() == 1, "the model was constructed once (on demand)");

    // 3. A second request is served by the loaded-model fast path (no re-load).
    ModelRouter::Grant again = router.route("m");
    CHECK(router.swap_count() == 1, "the second request did not re-load (loaded-model fast path)");
    CHECK(state.constructions_total() == 1, "no second construction (the fast path reuses the loaded model)");

    // Release both grants so no in-flight request holds the model (VRAM is actually freed on evict).
    serve.reset();
    again.reset();
    CHECK(router.in_flight() == 0, "no in-flight request holds the model");

    // 4. The TTL tick evicts the idle model (from the loaded state, not a fixed pre-resident).
    router.run_ttl_tick(std::chrono::steady_clock::now() + std::chrono::seconds(5));
    CHECK(router.loaded_id().empty(), "the idle model was evicted by the TTL tick");
    CHECK(state.destroyed_of("m") == 1, "the evicted model's backend was destroyed (VRAM freed)");

    // 5. The next request reloads it ON DEMAND.
    ModelRouter::Grant reload = router.route("m");
    CHECK(router.swap_count() == 2, "the reload was a fresh load/swap");
    CHECK(router.loaded_id() == "m", "the model was reloaded on the next request");
    CHECK(state.constructions_total() == 2, "the model was re-constructed on demand");
    CHECK(reload.backend != nullptr && reload.backend->id() == "m", "the reloaded grant is valid");
    (void)reload;
}

void test_8_readiness_timeout_rejects() {
    std::printf("[test] 8. route() rejects with std::runtime_error when the readiness gate times out\n");
    Config config;
    ModelConfig m;
    m.id = "slow";
    m.artifact = "/tmp/slow.ninfer";
    m.ttl = 0;
    config.models.push_back(m);
    config.global_ttl = 0;
    config.health_check_timeout = 1; // 1s gate so the readiness timeout is quick

    ModelRouter router(make_options(config), {},
                       [](const ModelConfig& model) -> std::unique_ptr<ModelBackend> {
                           return std::make_unique<NeverReadyBackend>(model.id);
                       });
    bool rejected = false;
    std::string message;
    try {
        (void)router.route("slow");
    } catch (const std::runtime_error& e) {
        rejected = true;
        message = e.what();
    }
    CHECK(rejected, "route() rejected the not-ready model (std::runtime_error)");
    CHECK(message.find("model is not ready") != std::string::npos,
          "the rejection message names the not-ready model");
}

void test_9_ttl_skips_while_in_flight() {
    std::printf("[test] 9. TTL tick skips eviction while a request is in-flight (pins the backend)\n");
    FakeState state;
    Config config;
    ModelConfig m;
    m.id = "m";
    m.artifact = "/tmp/m.ninfer";
    m.ttl = 2; // effective ttl 2s
    config.models.push_back(m);
    config.global_ttl = 0;
    config.health_check_timeout = 5;

    ModelRouter router(make_options(config), {}, make_factory(&state));

    // Load the model and hold the grant: an in-flight request pins the backend.
    ModelRouter::Grant held = router.route("m");
    CHECK(router.loaded_id() == "m", "the model is loaded and the grant is held");
    CHECK(router.in_flight() == 1, "the held grant is in-flight");

    // A tick far past the effective TTL must NOT evict while in-flight.
    router.run_ttl_tick(std::chrono::steady_clock::now() + std::chrono::seconds(10));
    CHECK(router.loaded_id() == "m", "the in-flight model was NOT evicted by the TTL tick");
    CHECK(router.resident() != nullptr, "the resident backend is still installed while in-flight");
    CHECK(state.destroyed_of("m") == 0, "the backend was not destroyed while a request is in-flight");

    // Release the grant: no in-flight request holds the model anymore.
    held.reset();
    CHECK(router.in_flight() == 0, "no in-flight request holds the model after release");

    // Now the TTL tick (still past TTL) evicts the idle model.
    router.run_ttl_tick(std::chrono::steady_clock::now() + std::chrono::seconds(10));
    CHECK(router.loaded_id().empty(), "the idle model was evicted once no request is in-flight");
    CHECK(state.destroyed_of("m") == 1, "the evicted model's backend was destroyed");
}

int main() {
    std::printf("ninfer model-router unit test\n");
    test_1_on_demand_first_load();
    test_2_load_from_empty_and_swap_back();
    test_3_concurrent_queueing();
    test_4_fifo_ordering();
    test_5_ttl_idle_unload();
    test_6_max_concurrency_gate();
    test_7_no_resident_lifecycle();
    test_8_readiness_timeout_rejects();
    test_9_ttl_skips_while_in_flight();

    if (g_failures == 0) {
        std::printf("\nALL TESTS PASSED\n");
        return 0;
    }
    std::printf("\n%d CHECK FAILURE(S)\n", g_failures);
    return 1;
}
