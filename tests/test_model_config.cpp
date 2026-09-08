#include "serve/model_config.h"

#include <iostream>
#include <string>

namespace {

using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    // --- load_config via fixture file ---
    std::string fixture = NINFER_SOURCE_DIR;
    fixture += "/tests/fixtures/serve-config.json";

    Config config = load_config(std::filesystem::path(fixture));

    // Model count.
    failures += check(config.models.size() == 5, "expected 5 models in fixture");

    // Model ids in expected order.
    if (config.models.size() == 5) {
        failures += check(config.models[0].id == "qwen3.6-27b/groupwise-int",
                         "first model id mismatch");
        failures += check(config.models[1].id == "qwen3.6-27b/nvfp4",
                         "second model id mismatch");
        failures += check(config.models[2].id == "qwen3.8-27b/groupwise-int",
                         "third model id mismatch");
        failures += check(config.models[3].id == "qwen3.8-27b/nvfp4",
                         "fourth model id mismatch");
        failures += check(config.models[4].id == "qwen3.6-35b-a3b/groupwise-int",
                         "fifth model id mismatch");
    }

    // Resident model is the first in file order.
    failures += check(config.resident().id == "qwen3.6-27b/groupwise-int",
                      "resident model is not the first model");

    // Every model has a non-empty artifact and identity == id.
    bool all_artifacts_nonempty = true;
    bool all_identity_eq_id     = true;
    for (const auto& model : config.models) {
        if (model.artifact.empty()) { all_artifacts_nonempty = false; }
        if (model.identity != model.id) { all_identity_eq_id = false; }
    }
    failures += check(all_artifacts_nonempty, "a model has an empty artifact");
    failures += check(all_identity_eq_id, "a model identity does not match its id");

    // Model lookup.
    failures += check(config.model("qwen3.8-27b/nvfp4") != nullptr,
                      "model lookup for qwen3.8-27b/nvfp4 returned null");
    failures += check(config.model("not-a-model") == nullptr,
                      "model lookup for unknown id returned non-null");

    // Top-level fields.
    failures += check(config.start_port == 5800, "start_port mismatch");
    failures += check(config.unload_timeout == 10, "unload_timeout mismatch");
    failures += check(config.global_ttl == 600, "global_ttl mismatch");

    // --- parse_config with inline JSON (single model) ---
    const std::string inline_json = R"({
        "models": {
            "test-model/foo": {
                "artifact": "out/test.ninfer",
                "identity": "test-model/foo",
                "ttl": 100,
                "name": "Test Model"
            }
        },
        "globalTTL": 100,
        "unloadTimeout": 5,
        "startPort": 9999
    })";
    Config single = parse_config(inline_json);
    failures += check(single.models.size() == 1, "inline parse produced wrong model count");
    failures += check(single.models[0].id == "test-model/foo",
                      "inline parse produced wrong model id");
    failures += check(single.models[0].artifact == "out/test.ninfer",
                      "inline parse did not preserve artifact");
    failures += check(single.start_port == 9999, "inline parse start_port mismatch");
    failures += check(single.global_ttl == 100, "inline parse global_ttl mismatch");

    // Missing `artifact` throws std::invalid_argument.
    bool missing_artifact_rejected = false;
    try {
        (void)parse_config(R"({"models": {"bad-model/x": {"identity": "bad-model/x"}}})");
    } catch (const std::invalid_argument&) {
        missing_artifact_rejected = true;
    }
    failures += check(missing_artifact_rejected, "model without artifact was accepted");

    // --- per-model engine overrides (the serve-config JSON fields the Nix config emits) ---
    const std::string overrides_json = R"({
        "models": {
            "ov/auto-kv": {
                "artifact": "out/auto.ninfer",
                "maxContext": 262144,
                "defaultMaxTokens": 32768,
                "kvCapacity": "auto",
                "kvDtype": "int8",
                "spec": "mtp",
                "draftTokens": 4,
                "lmHeadDraft": true,
                "prefillChunk": 4096,
                "vision": true
            },
            "ov/string-kv": {
                "artifact": "out/string.ninfer",
                "kvCapacity": "131072"
            }
        }
    })";
    const Config overrides = parse_config(overrides_json);
    const ModelConfig& auto_kv  = overrides.models[0];
    const ModelConfig& string_kv = overrides.models[1];
    failures += check(auto_kv.overrides.max_context.has_value() && *auto_kv.overrides.max_context == 262144,
                      "maxContext override not parsed");
    failures += check(auto_kv.overrides.default_max_tokens.has_value() &&
                          *auto_kv.overrides.default_max_tokens == 32768,
                      "defaultMaxTokens override not parsed");
    failures += check(auto_kv.overrides.kv_capacity.has_value() &&
                          auto_kv.overrides.kv_capacity->mode == ninfer::KvCapacityMode::Automatic,
                      "kvCapacity \"auto\" did not select automatic sizing");
    failures += check(auto_kv.overrides.kv_cache.has_value() &&
                          *auto_kv.overrides.kv_cache == ninfer::KvCacheStorage::Int8Group64,
                      "kvDtype int8 did not select Int8Group64");
    failures += check(auto_kv.overrides.speculative.has_value() &&
                          auto_kv.overrides.speculative->backend == ninfer::SpeculativeBackend::Mtp &&
                          auto_kv.overrides.speculative->draft_tokens == 4 &&
                          auto_kv.overrides.speculative->proposal_head == ninfer::ProposalHead::Optimized,
                      "spec/draftTokens/lmHeadDraft trio did not select Mtp+optimized head");
    failures += check(auto_kv.overrides.prefill_chunk.has_value() && *auto_kv.overrides.prefill_chunk == 4096,
                      "prefillChunk override not parsed");
    failures += check(auto_kv.overrides.enable_vision.has_value() && *auto_kv.overrides.enable_vision,
                      "vision override not parsed");
    // A numeric string kvCapacity (the Nix config emits it as a string) parses to explicit tokens.
    failures += check(string_kv.overrides.kv_capacity.has_value() &&
                          string_kv.overrides.kv_capacity->mode == ninfer::KvCapacityMode::Explicit &&
                          string_kv.overrides.kv_capacity->explicit_tokens == 131072,
                      "numeric-string kvCapacity did not parse to explicit tokens");
    // Absent fields stay nullopt (the engine falls back to the registered model defaults).
    failures += check(!string_kv.overrides.max_context && !string_kv.overrides.kv_cache &&
                          !string_kv.overrides.speculative && !string_kv.overrides.enable_vision,
                      "absent override fields are unexpectedly set");

    // --- per-model engine param validation (grounded in the engine's validate_target_options) ---
    auto reject_message = [](const std::string& json) -> std::string {
        try {
            (void)parse_config(json);
            return std::string(); // no throw
        } catch (const std::invalid_argument& e) {
            return e.what();
        }
    };

    // kvCapacity (explicit) < maxContext is rejected (ground: layouts_impl.h "kv_capacity must be
    // at least max_context").
    const std::string kv_low =
        reject_message(R"({"models":{"kv/low":{"artifact":"out/k.ninfer","maxContext":2048,"kvCapacity":1024}}})");
    failures += check(!kv_low.empty(), "kvCapacity < maxContext was accepted");
    failures += check(kv_low.find("kvCapacity") != std::string::npos,
                      "kvCapacity < maxContext rejection message does not mention kvCapacity");
    // kvCapacity (explicit) >= maxContext is accepted.
    const std::string kv_ok =
        reject_message(R"({"models":{"kv/ok":{"artifact":"out/k.ninfer","maxContext":2048,"kvCapacity":4096}}})");
    failures += check(kv_ok.empty(), "valid kvCapacity >= maxContext was rejected");
    // kvCapacity "auto" (automatic sizing) is exempt from the >= maxContext check.
    const std::string kv_auto =
        reject_message(R"({"models":{"kv/auto":{"artifact":"out/k.ninfer","maxContext":2048,"kvCapacity":"auto"}}})");
    failures += check(kv_auto.empty(), "kvCapacity \"auto\" (automatic sizing) was rejected");

    // prefillChunk that is not a multiple of 128 is rejected (ground: "nonzero multiple of 128").
    const std::string prefill_bad =
        reject_message(R"({"models":{"p/bad":{"artifact":"out/p.ninfer","prefillChunk":100}}})");
    failures += check(!prefill_bad.empty(), "prefillChunk not a multiple of 128 was accepted");
    failures += check(prefill_bad.find("prefillChunk") != std::string::npos,
                      "prefillChunk rejection message does not mention prefillChunk");
    // prefillChunk == 0 is rejected (nonzero required).
    const std::string prefill_zero =
        reject_message(R"({"models":{"p/zero":{"artifact":"out/p.ninfer","prefillChunk":0}}})");
    failures += check(!prefill_zero.empty(), "prefillChunk == 0 was accepted");
    // A valid multiple of 128 is accepted.
    const std::string prefill_ok =
        reject_message(R"({"models":{"p/ok":{"artifact":"out/p.ninfer","prefillChunk":256}}})");
    failures += check(prefill_ok.empty(), "valid prefillChunk (multiple of 128) was rejected");

    // speculative draftTokens out of the mtp range [1,5] is rejected (ground: speculative_options.h).
    const std::string mtp_high =
        reject_message(R"({"models":{"s/mtp":{"artifact":"out/s.ninfer","spec":"mtp","draftTokens":10}}})");
    failures += check(!mtp_high.empty(), "mtp draftTokens out of [1,5] was accepted");
    failures += check(mtp_high.find("draft") != std::string::npos,
                      "mtp draftTokens rejection message does not mention the draft window");
    // dflash draftTokens out of the [1,15] range is rejected.
    const std::string dflash_high =
        reject_message(R"({"models":{"s/dflash":{"artifact":"out/s.ninfer","spec":"dflash","draftTokens":20}}})");
    failures += check(!dflash_high.empty(), "dflash draftTokens out of [1,15] was accepted");
    // A valid dflash draftTokens (15) is accepted.
    const std::string dflash_ok =
        reject_message(R"({"models":{"s/dflash-ok":{"artifact":"out/s.ninfer","spec":"dflash","draftTokens":15}}})");
    failures += check(dflash_ok.empty(), "valid dflash draftTokens (15) was rejected");
    // draftTokens without a spec backend is rejected (the None case requires draft_tokens == 0).
    const std::string draft_no_spec =
        reject_message(R"({"models":{"s/nospec":{"artifact":"out/s.ninfer","draftTokens":3}}})");
    failures += check(!draft_no_spec.empty(), "draftTokens without a spec backend was accepted");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
