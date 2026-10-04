#include "models/qwen3_5/frontend/resources.h"

#include "artifact/schema.h"

namespace ninfer::models::qwen3_5 {

void parse_resources(FrontendResources& resources, const Config& config) {
    // GGUF artifacts are token-input-only: the four text resource objects are synthesized empty
    // (0 bytes), so there is no tokenizer sidecar. prepare_tokens/score_tokens/generate-with-
    // token-ids are supported; text-prompt (string) inputs are unavailable without a tokenizer.
    // Token-only guard: with an empty tokenizer_json, resources.tokenizer stays null,
    // public_token_count stays 0, and the vision-preprocessor, stop-token, and DFlash2 selector
    // checks are all skipped. The non-empty (.ninfer) path below is unchanged.
    if (resources.tokenizer_json.empty()) {
        return;
    }
    resources.tokenizer = std::make_shared<const frontend::Tokenizer>(
        frontend::TokenizerResources{resources.tokenizer_json, resources.tokenizer_config_json,
                                     resources.generation_config_json});
    const auto count = resources.tokenizer->vocab_size();
    if (!count || count > config.text.vocab_size ||
        !resources.tokenizer->has_exact_token_domain(count)) {
        throw artifact::ArtifactError(
            "tokenizer must expose a contiguous public domain within embedding rows");
    }
    resources.public_token_count = static_cast<std::uint32_t>(count);
    for (const auto token : resources.tokenizer->default_stop_token_ids()) {
        if (!resources.tokenizer->is_valid_token(token)) {
            throw artifact::ArtifactError("stop token is outside the public token domain");
        }
    }
    if (config.vision) {
        for (const auto bytes :
             {resources.preprocessor_config_json, resources.video_preprocessor_config_json}) {
            const auto value = artifact::parse_json(bytes, "Vision processor");
            if (!value.is_object() || !value.contains("patch_size") ||
                !value.contains("temporal_patch_size") || !value.contains("merge_size")) {
                throw artifact::ArtifactError("Vision processor is missing patch geometry");
            }
            if (artifact::require_u64(value.at("patch_size"), "patch_size", true) !=
                    config.vision->patch_size ||
                artifact::require_u64(value.at("temporal_patch_size"), "temporal_patch_size",
                                      true) != config.vision->temporal_patch_size ||
                artifact::require_u64(value.at("merge_size"), "merge_size", true) !=
                    config.vision->spatial_merge_size) {
                throw artifact::ArtifactError(
                    "Vision processor geometry differs from component config");
            }
        }
    }
    if (config.draft && config.draft->dflash2 && config.draft->dflash2->selector_top_k > count) {
        throw artifact::ArtifactError("DFlash2 selector_top_k exceeds public token domain");
    }
}

} // namespace ninfer::models::qwen3_5
