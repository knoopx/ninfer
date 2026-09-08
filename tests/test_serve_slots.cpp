// Host-side (no GPU) test for the GET /slots live slot state (llama-parity real-design §3, step 8).
// Verifies the pure slot-shaping function make_slots_list: the no-resident baseline (all slots
// idle), an injected in-flight count (a processing slot reports is_processing=true), the
// concurrency clamp, the ?model= filter (resident vs non-resident), and the is_swapping "loading"
// marker. No Engine, no router, no network -- the handler wires these scalar inputs from the live
// ModelRouter + resident backend (in_flight/loaded_id/is_swapping + runtime_stats); the shaping
// logic itself is what is exercised here, so it is host-testable without a GPU.

#include "serve/openai_common.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <string>
#include <vector>

namespace ns = ninfer::serve;
using ns::make_slots_list;
using Json = nlohmann::json;

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::fprintf(stderr, "FAIL: %s\n", message);
    return 1;
}

// A slot object must carry exactly these fields (no more, no less).
bool exact_fields(const Json& slot, const char* what) {
    const std::vector<std::string> keys = {"id", "model", "is_processing", "state"};
    if (slot.size() != keys.size()) {
        std::fprintf(stderr, "FAIL: %s: unexpected field count %zu\n", what, slot.size());
        return false;
    }
    for (const auto& key : keys) {
        if (!slot.contains(key)) {
            std::fprintf(stderr, "FAIL: %s: missing field '%s'\n", what, key.c_str());
            return false;
        }
    }
    return true;
}

// (a) no-resident / empty router: max_concurrency slots, all idle, empty model id.
int test_no_resident_all_idle() {
    int failures = 0;
    const Json slots = Json::parse(make_slots_list(/*max_concurrency=*/4, /*loaded_id=*/"",
                                                   /*model_filter=*/"", /*in_flight_total=*/0,
                                                   /*is_swapping=*/false, /*prefilling=*/0));
    failures += check(slots.is_array() && slots.size() == 4, "no-resident: expected 4 slots");
    for (std::size_t i = 0; i < slots.size(); ++i) {
        const Json& slot = slots[i];
        failures += check(exact_fields(slot, "no-resident slot"), "no-resident field set");
        failures += check(slot["id"].get<int>() == static_cast<int>(i),
                          "no-resident: slot id is its index");
        failures += check(slot["model"].is_string() && slot["model"].get<std::string>().empty(),
                          "no-resident: empty model id");
        failures += check(slot["is_processing"].get<bool>() == false, "no-resident: slot is idle");
        failures += check(slot["state"].get<std::string>() == "idle", "no-resident: state is idle");
    }
    return failures;
}

// (b) injected in-flight state on the resident: the first min(in_flight, max) slots process.
int test_in_flight_marks_processing() {
    int failures = 0;
    const std::string model = "qwen3.6-27b/groupwise-int";
    // 3 in-flight of 4 slots, 1 prefilling -> slots 0..2 process (state prefill), slot 3 idle.
    const Json slots =
        Json::parse(make_slots_list(4, model, "", /*in_flight_total=*/3, /*is_swapping=*/false,
                                    /*prefilling=*/1));
    failures += check(slots.size() == 4, "in-flight: expected 4 slots");
    for (std::size_t i = 0; i < slots.size(); ++i) {
        const Json& slot = slots[i];
        const bool expected_processing = i < 3;
        const std::string expected_state = expected_processing ? "prefill" : "idle";
        failures += check(slot["is_processing"].get<bool>() == expected_processing,
                          "in-flight: is_processing mismatch");
        failures += check(slot["model"].get<std::string>() == model,
                          "in-flight: slot carries the resident model");
        failures += check(slot["state"].get<std::string>() == expected_state,
                          "in-flight: state mismatch");
    }
    // A resident with in-flight but no prefill -> processing slots are in "decode".
    const Json decode =
        Json::parse(make_slots_list(2, model, "", /*in_flight_total=*/2, false, /*prefilling=*/0));
    failures += check(decode[0]["is_processing"].get<bool>() &&
                          decode[0]["state"].get<std::string>() == "decode",
                      "decode: a processing slot with no prefill is in decode");
    return failures;
}

// (c) the in-flight count is clamped to max_concurrency (extra generations cannot occupy slots).
int test_in_flight_clamped_to_concurrency() {
    int failures = 0;
    const Json slots =
        Json::parse(make_slots_list(2, "m", "", /*in_flight_total=*/5, false, /*prefilling=*/0));
    failures += check(slots.size() == 2, "clamp: slot count is max_concurrency, not in-flight");
    failures += check(slots[0]["is_processing"].get<bool>() && slots[1]["is_processing"].get<bool>(),
                      "clamp: both slots process when in-flight exceeds capacity");
    return failures;
}

// (d) a ?model= filter naming a registered-but-NON-resident model reports all slots idle.
int test_non_resident_filter_all_idle() {
    int failures = 0;
    const Json slots =
        Json::parse(make_slots_list(4, /*loaded_id=*/"resident", /*model_filter=*/"other",
                                    /*in_flight_total=*/3, false, /*prefilling=*/0));
    for (std::size_t i = 0; i < slots.size(); ++i) {
        const Json& slot = slots[i];
        failures += check(slot["model"].get<std::string>() == "other",
                          "filter: slot carries the filtered model");
        failures += check(slot["is_processing"].get<bool>() == false,
                          "filter: non-resident slot is idle");
        failures += check(slot["state"].get<std::string>() == "idle",
                          "filter: non-resident slot state is idle");
    }
    // A filter matching the resident reports its live in-flight state.
    const Json resident =
        Json::parse(make_slots_list(4, "resident", "resident", /*in_flight_total=*/2, false, 0));
    failures += check(resident[0]["is_processing"].get<bool>() &&
                          resident[1]["is_processing"].get<bool>(),
                      "filter: resident match marks its in-flight slots processing");
    failures += check(!resident[2]["is_processing"].get<bool>() &&
                          !resident[3]["is_processing"].get<bool>(),
                      "filter: resident match leaves surplus slots idle");
    return failures;
}

// (f) a swap in flight marks the resident's slots "loading" (informational state marker).
int test_swapping_marks_loading() {
    int failures = 0;
    const Json slots =
        Json::parse(make_slots_list(2, "resident", "", /*in_flight_total=*/0,
                                    /*is_swapping=*/true, /*prefilling=*/0));
    failures += check(slots[0]["state"].get<std::string>() == "loading" &&
                          slots[1]["state"].get<std::string>() == "loading",
                      "swapping: resident slots are marked loading");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_no_resident_all_idle();
    failures += test_in_flight_marks_processing();
    failures += test_in_flight_clamped_to_concurrency();
    failures += test_non_resident_filter_all_idle();
    failures += test_swapping_marks_loading();
    if (failures == 0) {
        std::printf("PASS: serve /slots (make_slots_list) -- all host-side checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "FAIL: %d serve /slots check(s) failed\n", failures);
    return 1;
}
