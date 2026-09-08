// Host-side (no GPU) test for the serving-owned ModelSseHub pub/sub (llama-parity real-design §5)
// and the ModelRouter status hook that drives it. No Engine, no model, no network -- a pure
// serving-state object plus a fake-backend router. Exercises:
//   (a) subscribe a fake client to the hub,
//   (b) drive a SYNTHETIC ModelStatusEvent through the hub (bypassing the real router) and assert
//       the fan-out reaches the client's queue in the exact wire shape,
//   (c) the current-status-on-connect record (the hub's current_status_record) reflects the router,
//   (d) a slow client whose bounded queue overflows is DROPPED without stalling the fan-out,
//   (e) disconnect (unsubscribe) tears down cleanly (no leak; a later publish no longer reaches it),
//   (f) the ModelRouter's status hook fires at a REAL status transition (a fake-backend router:
//       load/swap via route() -> "loading"/"loaded", and unload() -> "unloaded"), so the hook firing
//       is exercised host-side; only the model load itself is faked (a full in-service GPU transition
//       needs the real Engine, but the routing/transitions and the hook here are the real ones).
//
// Bare main() + a small CHECK macro, matching the other host tests; the binary exits non-zero on any
// failure.

#include "serve/model_sse_hub.h"
#include "serve/model_router.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ns = ninfer::serve;
using ns::ModelBackend;
using ns::ModelRouter;
using ns::ModelSseHub;
using ns::ModelStatusEvent;

namespace {

// ---- (f) a fake backend + a real ModelRouter (mirrors test_model_router.cpp) ----

struct FakeState {
    mutable std::mutex mu;
    std::unordered_map<std::string, int> constructed;
    // The "loading" gate: when closed, a non-resident fake blocks in its constructor (a slow load).
    std::atomic<bool> gate_closed{false};
    std::mutex gate_mu;
    std::condition_variable gate_cv;

    int constructions_total() const {
        std::lock_guard lock(mu);
        int total = 0;
        for (const auto& [_, count] : constructed) { total += count; }
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
            std::unique_lock<std::mutex> lock(state_->gate_mu);
            state_->gate_cv.wait(lock, [this] { return !this->state_->gate_closed.load(); });
        }
    }
    std::string id() const override { return id_; }
    bool is_available() const override { return true; } // ready once constructed (no async gate)
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
        static const ninfer::EngineOptions engine; // a stable dummy (the fake has no Engine)
        return engine;
    }

private:
    std::string id_;
    FakeState* state_;
    bool blocks_;
};

// A single-model Config (ttl 0 so the 1s TTL ticker is a no-op; the hook fires only on the driven
// transitions, keeping the test deterministic and single-threaded on the main thread).
ns::Config make_config() {
    ns::Config config;
    ns::ModelConfig res;
    res.id = "res";
    res.artifact = "/tmp/res.ninfer";
    res.ttl = 0;
    config.models.push_back(res);
    config.global_ttl = 0;
    config.health_check_timeout = 5;
    return config;
}

ModelRouter::BackendFactory make_factory(FakeState* state) {
    return [state](const ns::ModelConfig& model) -> std::unique_ptr<ModelBackend> {
        const bool blocks = state->gate_closed.load();
        return std::make_unique<FakeModelBackend>(model.id, state, blocks);
    };
}

ns::ServeOptions make_options(const ns::Config& config) {
    ns::ServeOptions options;
    options.model_config = config;
    options.artifact_path = "/tmp/default.ninfer";
    return options;
}

} // namespace

namespace {
int g_failures = 0;
}

#define CHECK(cond, msg)                                                          \
    do {                                                                          \
        if (cond) {                                                               \
            std::printf("  PASS: %s\n", msg);                                     \
        } else {                                                                  \
            std::printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);         \
            ++g_failures;                                                         \
        }                                                                         \
    } while (0)

// (a) + (b) subscribe a fake client, drive a synthetic ModelStatusEvent through the hub, and assert
// the fan-out reaches the client's queue in the exact wire shape.
void test_ab_subscribe_and_fanout() {
    std::printf("[test] a+b. subscribe a fake client; a synthetic event fans out to its queue\n");
    ModelSseHub hub;
    const auto sub = hub.subscribe();
    CHECK(hub.active(sub), "the subscribed client is active");

    // A synthetic event (bypassing the real router, design §5.1): the hub formats it into the exact
    // model_status record (the same shape current_status_record emits) and enqueues it on the client.
    hub.publish(ModelStatusEvent{"loading", "qwen3.6-27b/groupwise-int", 0.0, 0});
    std::string record;
    const auto got = hub.pop_blocking(sub, &record, std::chrono::milliseconds(10));
    CHECK(got == ModelSseHub::PopResult::Got, "the fan-out record reached the client");
    CHECK(record.find("\"event\":\"model_status\"") != std::string::npos,
          "the record is the model_status SSE event");
    CHECK(record.find("\"model\":\"qwen3.6-27b/groupwise-int\"") != std::string::npos,
          "the record carries the event's model");
    CHECK(record.find("\"status\":\"loading\"") != std::string::npos,
          "the record carries the event's status");
    CHECK(record.find("\"progress\":") != std::string::npos,
          "a loading record carries a progress field");

    // A quiet drain (no event) times out on the keepalive cadence (the writer would write a comment).
    const auto idle = hub.pop_blocking(sub, &record, std::chrono::milliseconds(10));
    CHECK(idle == ModelSseHub::PopResult::TimedOut, "an idle drain times out (keepalive window)");
    hub.unsubscribe(sub);
}

// (c) current-status-on-connect: the hub's connect record reflects the live router status.
void test_c_current_status_on_connect() {
    std::printf("[test] c. current-status-on-connect record reflects the live router\n");
    FakeState state;
    ModelRouter router(make_options(make_config()), {}, make_factory(&state));
    ModelSseHub hub;
    hub.bind(router);

    // No model loaded at startup (no-resident): the connect record is "unloaded".
    std::string record = hub.current_status_record();
    CHECK(record.find("\"status\":\"unloaded\"") != std::string::npos,
          "the connect record is 'unloaded' when nothing is loaded");

    // Load a model (a do_swap from the empty state): the connect record is now "loaded".
    ModelRouter::Grant g = router.route("res");
    record = hub.current_status_record();
    CHECK(record.find("\"status\":\"loaded\"") != std::string::npos,
          "the connect record is 'loaded' after a model is loaded");
    CHECK(record.find("\"model\":\"res\"") != std::string::npos,
          "the connect record carries the loaded model id");
    g.reset();
    (void)state;
}

// (d) a slow client (its bounded queue overflows) is DROPPED without stalling the fan-out: a
// second (timely) subscriber still receives events after the first is dropped.
void test_d_slow_client_drop() {
    std::printf("[test] d. a slow client's queue overflow drops it without stalling the fan-out\n");
    ModelSseHub hub;
    // The "slow" client gets a tiny queue (depth 1) and never drains it.
    const auto slow = hub.subscribe(/*queue_depth=*/1);
    hub.publish(ModelStatusEvent{"loading", "m", 0.0, 0}); // fills the depth-1 queue
    hub.publish(ModelStatusEvent{"loaded", "m", 0.0, 0});  // queue full -> the slow client is dropped

    // The slow client is now dropped: its next drain reports Inactive (the writer stops; the backlog
    // is discarded, so the too-slow client does not stall or drown the fan-out).
    std::string record;
    CHECK(hub.pop_blocking(slow, &record, std::chrono::milliseconds(10)) ==
              ModelSseHub::PopResult::Inactive,
          "the slow (dropped) client's drain reports Inactive (the fan-out did not stall)");
    CHECK(hub.active(slow), "a dropped subscriber stays registered until its writer unsubscribes");

    // A fresh, timely subscriber STILL receives the next event (the fan-out is not stalled by the
    // dropped slow client).
    const auto fast = hub.subscribe();
    hub.publish(ModelStatusEvent{"unloaded", "m", 0.0, 0});
    const auto got = hub.pop_blocking(fast, &record, std::chrono::milliseconds(10));
    CHECK(got == ModelSseHub::PopResult::Got &&
              record.find("\"status\":\"unloaded\"") != std::string::npos,
          "a timely subscriber still receives events after a slow client is dropped");

    hub.unsubscribe(slow);
    hub.unsubscribe(fast);
}

// (e) disconnect unsubscribes cleanly: after unsubscribe the client is no longer active and a later
// publish does not reach it (no leak; the subscriber is fully removed).
void test_e_unsubscribe_clean() {
    std::printf("[test] e. disconnect unsubscribes cleanly (no leak; publish no longer reaches it)\n");
    ModelSseHub hub;
    const auto sub = hub.subscribe();
    CHECK(hub.active(sub), "the subscriber is active before unsubscribe");
    hub.unsubscribe(sub);
    CHECK(!hub.active(sub), "the subscriber is gone after unsubscribe");

    // A publish after unsubscribe must not reach the removed client (its drain reports Inactive).
    hub.publish(ModelStatusEvent{"loaded", "m", 0.0, 0});
    std::string record;
    CHECK(hub.pop_blocking(sub, &record, std::chrono::milliseconds(10)) ==
              ModelSseHub::PopResult::Inactive,
          "a removed subscriber no longer receives fan-outs");
    // Unsubscribing again is a no-op (idempotent; the writer path does this on the drop path too).
    hub.unsubscribe(sub);
}

// (f) the ModelRouter's status hook fires at a REAL status transition: load a model via route()
// (a do_swap: loading -> loaded) and unload() it (unloaded). The hook is wired exactly as
// HttpServer::attach does (to a recording sink). Host-side (a fake backend, no Engine), so the
// routing/transitions and the hook are the real ones -- only the model load is faked.
void test_f_router_hook_fires_at_transition() {
    std::printf("[test] f. the ModelRouter status hook fires at real transitions (fake backend)\n");
    FakeState state;
    ModelRouter router(make_options(make_config()), {}, make_factory(&state));

    std::mutex mu;
    std::vector<ModelStatusEvent> events;
    auto record = [&](const ModelStatusEvent& ev) {
        std::lock_guard lock(mu);
        events.push_back(ev);
    };
    router.set_status_hook(record); // wired exactly as HttpServer::attach does

    // A real load (route -> do_swap from the empty state): fires "loading res" then "loaded res".
    ModelRouter::Grant g = router.route("res");
    CHECK(router.loaded_id() == "res", "route() loaded the model");
    {
        std::lock_guard lock(mu);
        CHECK(events.size() >= 2, "at least two events fired for the load transition");
        if (events.size() >= 2) {
            CHECK(events[0].status == "loading" && events[0].model == "res",
                  "the first event is 'loading res' (swap-start)");
            CHECK(events[1].status == "loaded" && events[1].model == "res",
                  "the second event is 'loaded res' (swap-complete)");
        }
    }

    // A real unload: fires "unloaded res".
    g.reset();
    router.unload();
    CHECK(router.loaded_id().empty(), "unload() cleared the resident");
    {
        std::lock_guard lock(mu);
        CHECK(!events.empty() && events.back().status == "unloaded" &&
                  events.back().model == "res",
              "the last event is 'unloaded res' (the explicit unload)");
    }
    (void)state;
}

int main() {
    std::printf("ninfer model-sse-hub unit test\n");
    test_ab_subscribe_and_fanout();
    test_c_current_status_on_connect();
    test_d_slow_client_drop();
    test_e_unsubscribe_clean();
    test_f_router_hook_fires_at_transition();

    if (g_failures == 0) {
        std::printf("\nALL TESTS PASSED\n");
        return 0;
    }
    std::printf("\n%d CHECK FAILURE(S)\n", g_failures);
    return 1;
}
