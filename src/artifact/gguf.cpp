#include "artifact/gguf.h"

#include "artifact/formats.h"
#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace ninfer::artifact {
namespace {

// ggml type id <-> QType. Bidirectional and total over the supported ggml ids: the float
// types reuse the existing .ninfer QTypes (F32 -> FP32, BF16 -> BF16, so no duplicate
// "fp32"/"bf16" format spellings); every quantized ggml type has its own QType value.
constexpr std::pair<std::uint32_t, QType> kGgmlFormats[] = {
    {0, QType::FP32},          {1, QType::GGUF_F16},
    {2, QType::GGUF_Q4_0},     {3, QType::GGUF_Q4_1},
    {6, QType::GGUF_Q5_0},     {7, QType::GGUF_Q5_1},
    {8, QType::GGUF_Q8_0},     {9, QType::GGUF_Q8_1},
    {10, QType::GGUF_Q2_K},    {11, QType::GGUF_Q3_K},
    {12, QType::GGUF_Q4_K},    {13, QType::GGUF_Q5_K},
    {14, QType::GGUF_Q6_K},    {15, QType::GGUF_Q8_K},
    {16, QType::GGUF_IQ2_XXS}, {17, QType::GGUF_IQ2_XS},
    {18, QType::GGUF_IQ3_XXS}, {19, QType::GGUF_IQ1_S},
    {20, QType::GGUF_IQ4_NL},  {21, QType::GGUF_IQ3_S},
    {22, QType::GGUF_IQ2_S},   {23, QType::GGUF_IQ4_XS},
    {24, QType::GGUF_I8},      {29, QType::GGUF_IQ1_M},
    {30, QType::BF16},         {41, QType::GGUF_Q1_0},
    {42, QType::GGUF_Q2_0},
};

std::uint64_t shape_elements(const Shape& shape) {
    std::uint64_t elements = 1;
    for (const auto dim : shape) { elements = checked_mul(elements, dim, "shape elements"); }
    return elements;
}

// Round-to-nearest-even F32 -> BF16 (the high 16 bits of the IEEE-754 word, rounded). Host-side
// cast for GGUF float vectors the engine represents as BF16; no CUDA dependency.
std::uint16_t f32_to_bf16(std::uint32_t bits) {
    return static_cast<std::uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

bool is_exact_fp32(std::string_view logical_name) {
    // The F32 tensors the model binds as exact FP32 (the GDN a_log / dt_bias vectors). Every
    // other F32 tensor in the GGUF (the norm vectors and the GDN convolution) the engine
    // represents as BF16, matching the reference GSQ-RCO artifact, so those are cast at load.
    return logical_name.ends_with("a_log") || logical_name.ends_with("dt_bias");
}

// GGUF v3 metadata value type ids (strata::MetaType).
constexpr std::uint32_t kMetaU8 = 0, kMetaI8 = 1, kMetaU16 = 2, kMetaI16 = 3, kMetaU32 = 4,
                        kMetaI32 = 5, kMetaF32 = 6, kMetaBool = 7, kMetaString = 8,
                        kMetaArray = 9, kMetaU64 = 10, kMetaI64 = 11, kMetaF64 = 12;

std::uint64_t gguf_scalar_bytes(std::uint32_t type) {
    switch (type) {
    case kMetaU8:
    case kMetaI8:
    case kMetaBool:
        return 1;
    case kMetaU16:
    case kMetaI16:
        return 2;
    case kMetaU32:
    case kMetaI32:
    case kMetaF32:
        return 4;
    case kMetaU64:
    case kMetaI64:
    case kMetaF64:
        return 8;
    default:
        throw ArtifactError("GGUF metadata type is not a fixed-width scalar");
    }
}

// A bounds-checked cursor over the mmapped header. strata's parsed metadata deliberately keeps only
// a 64-item sample of each array, so the full tokenizer vocabulary and merge list are re-read from
// the header here (the GGUF is the only tokenizer source a .gguf artifact carries).
class GgufHeaderCursor {
public:
    GgufHeaderCursor(const std::uint8_t* base, std::size_t size) : base_(base), size_(size) {}
    void need(std::uint64_t n) const {
        if (n > size_ || pos_ > size_ - static_cast<std::size_t>(n)) {
            throw ArtifactError("GGUF header is truncated");
        }
    }
    template <class T> T read() {
        need(sizeof(T));
        T value;
        std::memcpy(&value, base_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return value;
    }
    std::string str() {
        const auto n = read<std::uint64_t>();
        need(n);
        std::string value(reinterpret_cast<const char*>(base_ + pos_), static_cast<std::size_t>(n));
        pos_ += static_cast<std::size_t>(n);
        return value;
    }
    void skip(std::uint64_t n) {
        need(n);
        pos_ += static_cast<std::size_t>(n);
    }

private:
    const std::uint8_t* base_;
    std::size_t size_;
    std::size_t pos_ = 0;
};

// The tokenizer fields the GGUF carries. Empty `tokens` means the file has no embedded tokenizer,
// and the reader falls back to the empty token-input-only resources.
struct GgufTokenizer {
    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    std::vector<std::int32_t> token_types;
    std::string chat_template;
    std::string pad_token, bos_token, eos_token;
    std::uint64_t eos_id = 0;
    bool has_chat_template = false;
    bool has_eos_id        = false;
    bool add_bos           = false;
};

GgufTokenizer read_gguf_tokenizer(const std::uint8_t* base, std::size_t size) {
    GgufHeaderCursor cursor(base, size);
    if (cursor.read<std::uint32_t>() != 0x46554747u) { throw ArtifactError("not a GGUF file"); }
    const auto version = cursor.read<std::uint32_t>();
    if (version != 3) { throw ArtifactError("GGUF v" + std::to_string(version) + " is not v3"); }
    (void)cursor.read<std::uint64_t>();  // n_tensors
    const auto n_kv = cursor.read<std::uint64_t>();
    GgufTokenizer out;
    std::uint64_t bos = 0, eos = 0, pad = 0;
    bool has_bos = false, has_eos = false, has_pad = false;
    for (std::uint64_t i = 0; i < n_kv; ++i) {
        const auto key  = cursor.str();
        const auto type = cursor.read<std::uint32_t>();
        if (key == "tokenizer.ggml.tokens" || key == "tokenizer.ggml.merges") {
            const auto elem  = cursor.read<std::uint32_t>();
            const auto count = cursor.read<std::uint64_t>();
            if (elem != kMetaString) { throw ArtifactError(key + " is not an array of strings"); }
            auto& target = key == "tokenizer.ggml.tokens" ? out.tokens : out.merges;
            target.reserve(static_cast<std::size_t>(count));
            for (std::uint64_t j = 0; j < count; ++j) { target.push_back(cursor.str()); }
            continue;
        }
        if (key == "tokenizer.ggml.token_type") {
            const auto elem  = cursor.read<std::uint32_t>();
            const auto count = cursor.read<std::uint64_t>();
            if (elem != kMetaI32) { throw ArtifactError(key + " is not an array of I32"); }
            out.token_types.reserve(static_cast<std::size_t>(count));
            for (std::uint64_t j = 0; j < count; ++j) {
                out.token_types.push_back(cursor.read<std::int32_t>());
            }
            continue;
        }
        if (type == kMetaArray) {
            const auto elem  = cursor.read<std::uint32_t>();
            const auto count = cursor.read<std::uint64_t>();
            if (elem == kMetaString) {
                for (std::uint64_t j = 0; j < count; ++j) { (void)cursor.str(); }
            } else {
                cursor.skip(count * gguf_scalar_bytes(elem));
            }
            continue;
        }
        if (type == kMetaString) {
            const auto value = cursor.str();
            if (key == "tokenizer.chat_template") {
                out.chat_template     = value;
                out.has_chat_template = true;
            }
            continue;
        }
        if (type == kMetaBool) {
            const auto value = cursor.read<std::uint8_t>();
            if (key == "tokenizer.ggml.add_bos_token") { out.add_bos = value != 0; }
            continue;
        }
        if (type == kMetaU32 || type == kMetaU64) {
            const auto value = type == kMetaU32 ? std::uint64_t(cursor.read<std::uint32_t>())
                                                : cursor.read<std::uint64_t>();
            if (key == "tokenizer.ggml.bos_token_id") {
                bos     = value;
                has_bos = true;
            } else if (key == "tokenizer.ggml.eos_token_id") {
                eos     = value;
                has_eos = true;
            } else if (key == "tokenizer.ggml.padding_token_id") {
                pad     = value;
                has_pad = true;
            }
            continue;
        }
        cursor.skip(gguf_scalar_bytes(type));
    }
    const auto token_at = [&](std::uint64_t id) {
        return id < out.tokens.size() ? out.tokens[static_cast<std::size_t>(id)] : std::string{};
    };
    if (has_bos) { out.bos_token = token_at(bos); }
    if (has_eos) {
        out.eos_token = token_at(eos);
        out.eos_id    = eos;
        out.has_eos_id = true;
    }
    if (has_pad) { out.pad_token = token_at(pad); }
    return out;
}

// GGUF token types: 1 NORMAL, 2 UNKNOWN, 3 CONTROL, 4 USER_DEFINED, 5 UNUSED, 6 BYTE. CONTROL and
// USER_DEFINED tokens are the HF added tokens; only CONTROL ones are special (the USER_DEFINED
// thinking/tool markers are added tokens a skip-special decode still presents). UNUSED entries are
// the padding placeholders: they belong to neither and the reference tokenizer drops them.
bool gguf_added_token(std::int32_t type) { return type == 3 || type == 4; }
bool gguf_special_token(std::int32_t type) { return type == 3; }
bool gguf_vocab_token(std::int32_t type) { return type == 1 || type == 2 || type == 6; }

// The HF tokenizer.json the engine's tokenizer reads: a GPT-2 BPE model plus the added tokens. The
// pipeline fields are omitted (the architecture's fixed semantics apply), and every added-token
// entry carries the full flag set the tokenizer requires.
Json tokenizer_json(const GgufTokenizer& tokenizer) {
    Json vocab = Json::object();
    Json added = Json::array();
    for (std::size_t id = 0; id < tokenizer.tokens.size(); ++id) {
        const auto type = id < tokenizer.token_types.size() ? tokenizer.token_types[id] : 1;
        if (gguf_added_token(type)) {
            added.push_back(Json{{"id", id},
                                 {"content", tokenizer.tokens[id]},
                                 {"single_word", false},
                                 {"lstrip", false},
                                 {"rstrip", false},
                                 {"normalized", false},
                                 {"special", gguf_special_token(type)}});
        } else if (gguf_vocab_token(type)) {
            vocab[tokenizer.tokens[id]] = id;
        }
    }
    Json model;
    model["type"]   = "BPE";
    model["vocab"]  = std::move(vocab);
    model["merges"] = tokenizer.merges;
    Json root;
    root["model"]        = std::move(model);
    root["added_tokens"] = std::move(added);
    return root;
}

Json tokenizer_config_json(const GgufTokenizer& tokenizer) {
    Json root;
    root["add_bos_token"]    = tokenizer.add_bos;
    root["add_prefix_space"] = false;
    if (!tokenizer.pad_token.empty()) { root["pad_token"] = tokenizer.pad_token; }
    if (!tokenizer.bos_token.empty()) { root["bos_token"] = tokenizer.bos_token; }
    if (!tokenizer.eos_token.empty()) { root["eos_token"] = tokenizer.eos_token; }
    Json decoder = Json::object();
    for (std::size_t id = 0; id < tokenizer.tokens.size(); ++id) {
        const auto type = id < tokenizer.token_types.size() ? tokenizer.token_types[id] : 1;
        if (!gguf_added_token(type)) { continue; }
        decoder[std::to_string(id)] = Json{{"content", tokenizer.tokens[id]},
                                          {"single_word", false},
                                          {"lstrip", false},
                                          {"rstrip", false},
                                          {"normalized", false},
                                          {"special", gguf_special_token(type)}};
    }
    root["added_tokens_decoder"] = std::move(decoder);
    return root;
}

Json generation_config_json(const GgufTokenizer& tokenizer) {
    Json root;
    // The GGUF carries one eos id; an absent one still needs a non-empty stop set for the loader.
    root["eos_token_id"] = tokenizer.has_eos_id ? tokenizer.eos_id : 0;
    return root;
}

const char* meta_type_name(strata::MetaType type) {
    switch (type) {
    case strata::MetaType::U8:
        return "U8";
    case strata::MetaType::I8:
        return "I8";
    case strata::MetaType::U16:
        return "U16";
    case strata::MetaType::I16:
        return "I16";
    case strata::MetaType::U32:
        return "U32";
    case strata::MetaType::I32:
        return "I32";
    case strata::MetaType::F32:
        return "F32";
    case strata::MetaType::BOOL:
        return "BOOL";
    case strata::MetaType::STRING:
        return "STRING";
    case strata::MetaType::ARRAY:
        return "ARRAY";
    case strata::MetaType::U64:
        return "U64";
    case strata::MetaType::I64:
        return "I64";
    case strata::MetaType::F64:
        return "F64";
    }
    throw ArtifactError("unknown GGUF metadata type");
}

// One GGUF KV value as JSON. Arrays longer than 64 elements are summarized as
// {"_array": "<type>", "count": N}: strata keeps at most the first 64 items, and the
// tokenizer merges/tokens arrays carry hundreds of thousands of strings.
Json meta_value_json(const strata::MetaValue& value) {
    switch (value.type) {
    case strata::MetaType::STRING:
        return Json(value.s);
    case strata::MetaType::ARRAY: {
        if (value.count > 64) {
            return Json{{"_array", meta_type_name(value.elem)}, {"count", value.count}};
        }
        Json array = Json::array();
        for (const auto& item : value.items) {
            if (item.type == strata::MetaType::STRING) {
                array.push_back(item.s);
            } else if (item.type == strata::MetaType::F32 || item.type == strata::MetaType::F64) {
                array.push_back(item.f);
            } else {
                array.push_back(item.u);
            }
        }
        return array;
    }
    case strata::MetaType::F32:
    case strata::MetaType::F64:
        return Json(value.f);
    case strata::MetaType::BOOL:
        return Json(value.u ? 1 : 0);
    case strata::MetaType::U8:
    case strata::MetaType::I8:
    case strata::MetaType::U16:
    case strata::MetaType::I16:
    case strata::MetaType::U32:
    case strata::MetaType::I32:
    case strata::MetaType::U64:
    case strata::MetaType::I64:
        return Json(value.u);
    }
    throw ArtifactError("unknown GGUF metadata value type");
}

// tokenizer.* keys keep only a summary so multi-MB tokenizer blobs stay out of the
// directory: scalars as-is, strings as a length summary, arrays as a count summary.
Json tokenizer_value_json(const strata::MetaValue& value) {
    switch (value.type) {
    case strata::MetaType::STRING:
        return Json{{"_type", "STRING"}, {"length", value.s.size()}};
    case strata::MetaType::ARRAY:
        return Json{{"_array", meta_type_name(value.elem)}, {"count", value.count}};
    case strata::MetaType::F32:
    case strata::MetaType::F64:
        return Json(value.f);
    case strata::MetaType::BOOL:
        return Json(value.u ? 1 : 0);
    case strata::MetaType::U8:
    case strata::MetaType::I8:
    case strata::MetaType::U16:
    case strata::MetaType::I16:
    case strata::MetaType::U32:
    case strata::MetaType::I32:
    case strata::MetaType::U64:
    case strata::MetaType::I64:
        return Json(value.u);
    }
    throw ArtifactError("unknown GGUF metadata value type");
}

Json metadata_json(const strata::GgufModel& model) {
    Json metadata = Json::object();
    for (const auto& [key, value] : model.meta().metadata()) {
        metadata[key] = key.rfind("tokenizer.", 0) == 0 ? tokenizer_value_json(value)
                                                         : meta_value_json(value);
    }
    return metadata;
}

const strata::MetaValue* meta_optional(const strata::GgufModel& model, const char* key) {
    const auto& values = model.meta().metadata();
    const auto found    = values.find(key);
    return found == values.end() ? nullptr : &found->second;
}

const strata::MetaValue* meta(const strata::GgufModel& model, const char* key) {
    const auto* value = meta_optional(model, key);
    if (!value) { throw ArtifactError(std::string("GGUF metadata is missing ") + key); }
    return value;
}

std::uint64_t meta_u64(const strata::GgufModel& model, const char* key) {
    const auto* value = meta(model, key);
    if (!value->is_num()) { throw ArtifactError(std::string(key) + " is not a number"); }
    return value->u;
}

double meta_double(const strata::GgufModel& model, const char* key) {
    const auto* value = meta(model, key);
    if (!value->is_num()) { throw ArtifactError(std::string(key) + " is not a number"); }
    return value->num();
}

// The NInfer TextConfig JSON (the src/models/qwen3_5/config.cpp schema) derived from the
// GGUF qwen35.* keys and tensor shapes. Every value is grounded in the file; a value that
// cannot be grounded is refused with a precise error, never invented.
Json text_config(const strata::GgufModel& model) {
    const auto hidden      = meta_u64(model, "qwen35.embedding_length");
    const auto block_count = meta_u64(model, "qwen35.block_count");
    const auto nextn       =
        meta_optional(model, "qwen35.nextn_predict_layers")
            ? meta_u64(model, "qwen35.nextn_predict_layers")
            : 0;
    if (!block_count || nextn >= block_count) {
        throw ArtifactError("qwen35.block_count and qwen35.nextn_predict_layers disagree");
    }
    const auto num_layers = block_count - nextn;

    // vocab_size from the embedding tensor (GGUF dim1), cross-checked with the tokenizer.
    const auto* embedding = model.find("token_embd.weight");
    if (!embedding || embedding->shape.size() != 2 || !embedding->shape[0]) {
        throw ArtifactError("token_embd.weight is missing or not a matrix");
    }
    const auto vocab = embedding->shape[1];
    if (embedding->shape[0] != hidden) {
        throw ArtifactError("token_embd rows differ from qwen35.embedding_length");
    }
    if (const auto* tokens = meta_optional(model, "tokenizer.ggml.token_type");
        tokens && tokens->type == strata::MetaType::ARRAY && tokens->count != vocab) {
        throw ArtifactError("tokenizer.ggml.token_type count differs from the embedding rows");
    }

    const auto interval = meta_u64(model, "qwen35.full_attention_interval");
    if (!interval) { throw ArtifactError("qwen35.full_attention_interval must be positive"); }
    Json layer_types = Json::array();
    std::uint64_t full = 0, linear = 0;
    for (std::uint64_t i = 0; i < num_layers; ++i) {
        const bool is_full = (i + 1) % interval == 0;
        layer_types.push_back(is_full ? "full_attention" : "linear_attention");
        if (is_full) {
            ++full;
        } else {
            ++linear;
        }
    }
    // Cross-check the per-block recurrent flag when present (its first entry is block 0).
    if (const auto* recurrent = meta_optional(model, "qwen35.attention.recurrent_layers");
        recurrent && recurrent->type == strata::MetaType::ARRAY && !recurrent->items.empty() &&
        recurrent->items.front().u != 0 && 1 % interval == 0) {
        throw ArtifactError(
            "qwen35.attention.recurrent_layers[0] marks block 0 full-attention, but "
            "qwen35.full_attention_interval says it is recurrent");
    }

    Json config = Json::object();
    config["architectures"]           = Json::array({"Qwen3_5ForCausalLM"});
    config["model_type"]              = "qwen3_5_text";
    config["hidden_size"]             = hidden;
    config["vocab_size"]              = vocab;
    config["num_hidden_layers"]       = num_layers;
    config["max_position_embeddings"] = meta_u64(model, "qwen35.context_length");
    // The file carries a separate output tensor, so the word embeddings are not tied.
    config["tie_word_embeddings"] = model.find("output.weight") == nullptr;
    config["rms_norm_eps"] = meta_double(model, "qwen35.attention.layer_norm_rms_epsilon");
    config["layer_types"]  = layer_types;
    config["intermediate_size"] = meta_u64(model, "qwen35.feed_forward_length");

    if (full || nextn) {
        const auto heads    = meta_u64(model, "qwen35.attention.head_count");
        const auto kv_heads = meta_u64(model, "qwen35.attention.head_count_kv");
        if (!kv_heads || heads % kv_heads) {
            throw ArtifactError("qwen35.attention head counts disagree");
        }
        // head_dim is the per-head attention width (qwen35.attention.key_length,
        // cross-checked equal to value_length); the rotary width the metadata carries
        // (qwen35.rope.dimension_count) is narrower and is reproduced from head_dim via
        // the partial_rotary_factor below.
        const auto head_dim = meta_u64(model, "qwen35.attention.key_length");
        if (head_dim != meta_u64(model, "qwen35.attention.value_length")) {
            throw ArtifactError("qwen35.attention key and value lengths disagree");
        }
        if (!head_dim) { throw ArtifactError("qwen35.attention.key_length must be positive"); }
        const auto rope_dim = meta_u64(model, "qwen35.rope.dimension_count");
        if (!rope_dim || rope_dim > head_dim) {
            throw ArtifactError("qwen35.rope.dimension_count must be positive and <= head_dim");
        }
        // The metadata carries the rotary width itself, so the partial_rotary_factor that
        // reproduces it from head_dim is rope_dim / head_dim.
        const double partial_rotary_factor = double(rope_dim) / double(head_dim);
        const auto* sections_raw = meta(model, "qwen35.rope.dimension_sections");
        if (sections_raw->type != strata::MetaType::ARRAY) {
            throw ArtifactError("qwen35.rope.dimension_sections is not an array");
        }
        std::vector<std::uint64_t> sections;
        for (const auto& item : sections_raw->items) { sections.push_back(item.u); }
        while (sections.size() > 1 && sections.back() == 0) { sections.pop_back(); }
        if (sections.size() != 3) {
            throw ArtifactError("qwen35.rope.dimension_sections must carry three sections");
        }
        std::uint64_t section_sum = 0;
        for (const auto section : sections) { section_sum += section; }
        if (section_sum * 2 != rope_dim) {
            throw ArtifactError("MRoPE sections differ from the rotary width");
        }
        Json mrope = Json::array();
        for (const auto section : sections) { mrope.push_back(section); }
        config["num_attention_heads"] = heads;
        config["num_key_value_heads"] = kv_heads;
        config["head_dim"]            = head_dim;
        config["rope_parameters"]     = Json{
            {"rope_theta", meta_double(model, "qwen35.rope.freq_base")},
            {"partial_rotary_factor", partial_rotary_factor},
            {"mrope_section", mrope},
        };
    }

    if (linear) {
        // GDN geometry from the ssm metadata: the key heads are the ssm groups, the value
        // heads are the time-step rank, and both share the state head dim. The value width
        // is the inner ssm projection width.
        const auto key_heads   = meta_u64(model, "qwen35.ssm.group_count");
        const auto value_heads = meta_u64(model, "qwen35.ssm.time_step_rank");
        const auto head_dim    = meta_u64(model, "qwen35.ssm.state_size");
        const auto conv_kernel = meta_u64(model, "qwen35.ssm.conv_kernel");
        if (!key_heads || !value_heads || !head_dim || !conv_kernel) {
            throw ArtifactError("qwen35.ssm dimensions must be positive");
        }
        const auto key_width   = checked_mul(key_heads, head_dim, "GDN key width");
        const auto value_width = checked_mul(value_heads, head_dim, "GDN value width");
        // The inner ssm projection is value-shaped.
        if (const auto* inner = meta_optional(model, "qwen35.ssm.inner_size");
            inner && inner->is_num() && inner->u != value_width) {
            throw ArtifactError("qwen35.ssm.inner_size disagrees with the GDN value width");
        }
        // The convolution channels come from a ssm_conv1d tensor ([kernel, channels]);
        // the channels must equal 2 * key_width + value_width (query, key, value order).
        const strata::TensorInfo* conv = nullptr;
        for (std::size_t s = 0; s < model.size() && !conv; ++s) {
            for (const auto& tensor : model.shard(s).tensors()) {
                if (tensor.name.ends_with(".ssm_conv1d.weight")) {
                    conv = &tensor;
                    break;
                }
            }
        }
        if (!conv || conv->shape.size() != 2) {
            throw ArtifactError("a ssm_conv1d.weight tensor is required for the GDN geometry");
        }
        if (conv->shape[0] != conv_kernel) {
            throw ArtifactError("ssm_conv1d kernel differs from qwen35.ssm.conv_kernel");
        }
        if (conv->shape[1] !=
            checked_add(checked_mul(2, key_width, "GDN key width"), value_width,
                        "GDN convolution channels")) {
            throw ArtifactError("ssm_conv1d channels disagree with the GDN key/value widths");
        }
        config["linear_num_key_heads"]   = key_heads;
        config["linear_key_head_dim"]    = head_dim;
        config["linear_num_value_heads"] = value_heads;
        config["linear_value_head_dim"]  = head_dim;
        config["linear_conv_kernel_dim"] = conv_kernel;
    }
    return config;
}

} // namespace

QType gguf_qtype(std::uint32_t ggml_type_id) {
    for (const auto& [ggml, format] : kGgmlFormats) {
        if (ggml == ggml_type_id) { return format; }
    }
    throw ArtifactError("unsupported ggml type " + std::to_string(ggml_type_id) + " (" +
                        strata::ggml_type_name(ggml_type_id) + ")");
}

std::uint32_t ggml_type(QType format) {
    for (const auto& [ggml, f] : kGgmlFormats) {
        if (f == format) { return ggml; }
    }
    throw ArtifactError("not a GGUF tensor format");
}

std::vector<std::string> logical_names(const std::string& gguf_name, std::size_t mtp_block) {
    if (gguf_name.rfind("blk.", 0) != 0) {
        if (gguf_name == "token_embd.weight") { return {"text/token_embedding"}; }
        if (gguf_name == "output.weight") { return {"text/output_head"}; }
        if (gguf_name == "output_norm.weight") { return {"text/final_norm"}; }
        return {};
    }
    // "blk.{i}." -> block index i and the tensor remainder.
    const auto dot = gguf_name.find('.', 4);
    if (dot == std::string::npos) { return {}; }
    std::size_t block = 0;
    for (const char digit : gguf_name.substr(4, dot - 4)) {
        if (digit < '0' || digit > '9') { return {}; }
        block = block * 10 + static_cast<std::size_t>(digit - '0');
    }
    const auto rest = gguf_name.substr(dot + 1);

    if (block == mtp_block) {
        const std::string m = "mtp/layers/0/";
        if (rest == "nextn.eh_proj.weight") { return {"mtp/input_projection"}; }
        if (rest == "nextn.enorm.weight") { return {"mtp/embedding_norm"}; }
        if (rest == "nextn.hnorm.weight") { return {"mtp/hidden_norm"}; }
        if (rest == "nextn.shared_head_norm.weight") { return {"mtp/final_norm"}; }
        if (rest == "attn_norm.weight") { return {m + "input_norm"}; }
        if (rest == "post_attention_norm.weight") { return {m + "post_attention_norm"}; }
        if (rest == "attn_q.weight") {
            // One combined tensor: query and gate are element-row sub-ranges (the .ninfer
            // binding splits them with Part ranges).
            return {m + "attention/query", m + "attention/gate"};
        }
        if (rest == "attn_k.weight") { return {m + "attention/key"}; }
        if (rest == "attn_v.weight") { return {m + "attention/value"}; }
        if (rest == "attn_output.weight") { return {m + "attention/output"}; }
        if (rest == "attn_q_norm.weight") { return {m + "attention/query_norm"}; }
        if (rest == "attn_k_norm.weight") { return {m + "attention/key_norm"}; }
        if (rest == "ffn_gate.weight") { return {m + "mlp/gate"}; }
        if (rest == "ffn_up.weight") { return {m + "mlp/up"}; }
        if (rest == "ffn_down.weight") { return {m + "mlp/down"}; }
        return {};
    }

    const auto p = "text/layers/" + std::to_string(block) + "/";
    if (rest == "attn_norm.weight") { return {p + "input_norm"}; }
    if (rest == "post_attention_norm.weight") { return {p + "post_attention_norm"}; }
    if (rest == "ffn_gate.weight") { return {p + "mlp/gate"}; }
    if (rest == "ffn_up.weight") { return {p + "mlp/up"}; }
    if (rest == "ffn_down.weight") { return {p + "mlp/down"}; }
    if (rest == "attn_q.weight") {
        // One combined tensor: query and gate are element-row sub-ranges (the .ninfer
        // binding splits them with Part ranges).
        return {p + "attention/query", p + "attention/gate"};
    }
    if (rest == "attn_k.weight") { return {p + "attention/key"}; }
    if (rest == "attn_v.weight") { return {p + "attention/value"}; }
    if (rest == "attn_output.weight") { return {p + "attention/output"}; }
    if (rest == "attn_q_norm.weight") { return {p + "attention/query_norm"}; }
    if (rest == "attn_k_norm.weight") { return {p + "attention/key_norm"}; }
    if (rest == "ssm_a") { return {p + "gdn/a_log"}; }
    if (rest == "ssm_dt.bias") { return {p + "gdn/dt_bias"}; }
    if (rest == "ssm_conv1d.weight") { return {p + "gdn/convolution"}; }
    if (rest == "ssm_alpha.weight") { return {p + "gdn/a_projection"}; }
    if (rest == "ssm_beta.weight") { return {p + "gdn/b_projection"}; }
    if (rest == "ssm_norm.weight") { return {p + "gdn/norm"}; }
    if (rest == "ssm_out.weight") { return {p + "gdn/output"}; }
    // One combined tensor: query/key/value are element-row sub-ranges of this tensor,
    // not separate whole tensors (the .ninfer binding splits them with Part ranges).
    if (rest == "attn_qkv.weight") { return {p + "gdn/query", p + "gdn/key", p + "gdn/value"}; }
    if (rest == "attn_gate.weight") { return {p + "gdn/z"}; }
    return {};
}

struct GgufReader::Impl {
    strata::GgufModel model;
    Directory directory;
    std::vector<std::optional<WeightGeometry>> geometries;
    std::vector<const std::uint8_t*> shard_bases;
    std::vector<std::uint64_t> data_starts;
    std::vector<std::uint64_t> logical_begins;
    std::vector<std::uint64_t> payload_sizes;
    std::size_t mtp_block = std::numeric_limits<std::size_t>::max();
    std::uint64_t file_bytes = 0;
    ArtifactId id{};
    // Owned F32->BF16 cast payloads for the GGUF float norm vectors (see is_norm_vector). The
    // cast bytes are not in any shard, so they live here and are served via read_object; the
    // map keys are TensorObject indices into directory.objects.
    std::vector<std::byte> cast_payload_;
    std::unordered_map<std::size_t, std::uint64_t> cast_offsets_;
    // Owned synthesized frontend resources: the GGUF's embedded tokenizer rendered as the HF
    // artifact strings the frontend reads. Every resource object points into this buffer.
    std::vector<std::byte> resources_payload_;

    explicit Impl(const std::filesystem::path& path) : model(strata::GgufModel::open(path.string())) {
        build_shards();
        build_objects();
        build_files();
        build_components();
        build_bindings();
        build_uses();
        build_resources();
        directory.metadata = metadata_json(model);
        Json provenance    = {
            {"source", "gguf_v3"},
            {"path", path.string()},
            {"shards", model.size()},
            {"tensors", directory.objects.size()},
        };
        if (const auto* architecture = meta_optional(model, "general.architecture")) {
            provenance["architecture"] = architecture->s;
        }
        directory.provenance = provenance;
        geometries.resize(directory.objects.size());
    }

private:
    void build_shards() {
        const auto shards = model.size();
        data_starts.resize(shards);
        logical_begins.resize(shards);
        payload_sizes.resize(shards);
        shard_bases.resize(shards);
        std::uint64_t logical = 0;
        for (std::size_t s = 0; s < shards; ++s) {
            const auto& shard = model.shard(s);
            data_starts[s]    = shard.data_start();
            payload_sizes[s]  = shard.file_size() - shard.data_start();
            logical_begins[s] = logical;
            logical           = checked_add(logical, payload_sizes[s], "logical payload");
            file_bytes        = checked_add(file_bytes, shard.file_size(), "file bytes");
            // The mmapped shard base, recovered from any tensor: tensor_data(t) is
            // base + data_start + t.offset. A metadata-only split shard has no tensors
            // and exposes no base; reads into it are refused.
            const auto& tensors = shard.tensors();
            shard_bases[s] = tensors.empty()
                                 ? nullptr
                                 : shard.tensor_data(tensors.front()) - data_starts[s] -
                                       tensors.front().offset;
        }
        directory.payload_bytes = logical;
        const auto block_count = meta_optional(model, "qwen35.block_count")
                                     ? meta_u64(model, "qwen35.block_count")
                                     : 0;
        const auto nextn = meta_optional(model, "qwen35.nextn_predict_layers")
                               ? meta_u64(model, "qwen35.nextn_predict_layers")
                               : 0;
        if (nextn && nextn < block_count) { mtp_block = block_count - 1; }
    }

    void build_objects() {
        struct Entry {
            std::size_t shard;
            const strata::TensorInfo* tensor;
        };
        std::vector<Entry> entries;
        for (std::size_t s = 0; s < model.size(); ++s) {
            for (const auto& tensor : model.shard(s).tensors()) {
                entries.push_back({s, &tensor});
            }
        }
        std::stable_sort(entries.begin(), entries.end(),
                         [&](const auto& a, const auto& b) {
                             return logical_begins[a.shard] + a.tensor->offset <
                                    logical_begins[b.shard] + b.tensor->offset;
                         });
        for (const auto& entry : entries) {
            const auto& tensor = *entry.tensor;
            const auto bytes   = strata::tensor_payload_bytes(tensor);
            if (!bytes) {
                throw ArtifactError("tensor " + tensor.name + " has no known payload size (ggml "
                                    "type " + std::to_string(tensor.type) + ")");
            }
            const auto names = logical_names(tensor.name, mtp_block);
            TensorObject object;
            object.id = tensor.name;
            // GGUF dim0 varies fastest (the contiguous K axis): reverse into NInfer order.
            // The GDN convolution (logical name "gdn/convolution", GGUF ".ssm_conv1d.weight")
            // is the one tensor whose model row dim is the GGUF fastest dim (kernel), so it is
            // NOT reversed: the model requests {kernel, channels} in the raw GGUF [kernel,
            // channels] order, while every other tensor's model row dim is the GGUF slowest dim.
            if (tensor.name.ends_with(".ssm_conv1d.weight")) {
                object.shape.assign(tensor.shape.begin(), tensor.shape.end());
            } else {
                object.shape.assign(tensor.shape.rbegin(), tensor.shape.rend());
            }
            const auto elements = shape_elements(object.shape);
            // GGUF float tensors are direct (unencoded) values, not raw block encodings.
            // F32 becomes an owned BF16 payload (the Op contract and the reference GSQ-RCO
            // artifact) unless the model binds it as exact FP32 (gdn/a_log, gdn/dt_bias), and
            // BF16 is already the engine's representation. Both stay Contiguous so the direct
            // consumers (weight_tensor, the BF16 direct ops) accept them; only the quantized
            // block encodings use the GgufNative layout.
            const bool cast = tensor.type == 0 &&
                              !std::any_of(names.begin(), names.end(), is_exact_fp32);
            // Q2_K (ggml type 10) has no Strata kernel in either GGUF route; the dispatch runs it
            // through the kernel ported from llama.cpp (see gguf_kernels.cu).
            const bool direct_float = tensor.type == 0 || tensor.type == 30;
            if (cast) {
                const auto* source = model.shard(entry.shard).tensor_data(tensor);
                const auto begin   = cast_payload_.size();
                const auto bf16_bytes = checked_mul(elements, 2, "cast payload bytes");
                cast_payload_.resize(begin + bf16_bytes);
                for (std::uint64_t i = 0; i < elements; ++i) {
                    std::uint32_t bits;
                    std::memcpy(&bits, source + i * 4, 4);
                    const std::uint16_t bf16 = f32_to_bf16(bits);
                    std::memcpy(cast_payload_.data() + begin + i * 2, &bf16, 2);
                }
                cast_offsets_.emplace(directory.objects.size(), begin);
                object.format = std::string(format_name(QType::BF16));
                object.layout = std::string(layout_name(QuantLayout::Contiguous));
                object.offset = begin;
                object.bytes  = bf16_bytes;
            } else if (direct_float) {
                object.format = std::string(format_name(gguf_qtype(tensor.type)));
                object.layout = std::string(layout_name(QuantLayout::Contiguous));
                object.offset =
                    checked_add(logical_begins[entry.shard], tensor.offset, "object offset");
                object.bytes = bytes;
            } else {
                object.format = std::string(format_name(gguf_qtype(tensor.type)));
                object.layout = "gguf_native";
                object.offset =
                    checked_add(logical_begins[entry.shard], tensor.offset, "object offset");
                object.bytes = bytes;
            }
            const auto handle = ObjectHandle{directory.objects.size()};
            directory.objects.push_back(std::move(object));
            directory.object_index.emplace(tensor.name, handle);
            for (const auto& alias : names) {
                directory.object_index.emplace(alias, handle);
            }
        }
    }

    void build_files() {
        for (std::size_t s = 0; s < model.size(); ++s) {
            FileRecord file;
            if (s) { file.path = std::filesystem::path(model.shard(s).path()).filename().string(); }
            file.payload_bytes = payload_sizes[s];
            file.logical_begin = logical_begins[s];
            directory.files.push_back(std::move(file));
        }
    }

    void build_components() {
        Component text;
        text.config = text_config(model);
        if (mtp_block != std::numeric_limits<std::size_t>::max()) {
            // The MTP draft (proposal) head is the shared LM head: declare a full (non-indexed)
            // proposal so the Optimized proposal-head path (lmHeadDraft) resolves to it.
            text.proposal = Proposal{};
        }
        directory.components.emplace("text", std::move(text));
        if (mtp_block != std::numeric_limits<std::size_t>::max()) {
            Component mtp;
            mtp.config = Json{{"architectures", Json::array({"Qwen3_5MTP"})}};
            mtp.target = "text";
            directory.components.emplace("mtp", std::move(mtp));
        }
    }

    // Parts-based parameter bindings for every logical name the model binds. A tensor whose
    // logical_names() yields one name gets a whole-object binding (one part over the object's
    // element domain). The combined qkv / q tensors (multiple names) get parts bindings: one
    // Part row-sub-range per logical name, with element offsets = row * hidden. The mixer
    // widths are derived from the text config built in build_components (never hardcoded to a
    // file's numbers): full-attention query width and GDN key/value widths.
    void build_bindings() {
        const auto& config = directory.components.at("text").config;
        std::uint64_t query_width = 0, key_width = 0, value_width = 0;
        if (config.contains("head_dim") && config.contains("num_attention_heads")) {
            query_width = checked_mul(config["num_attention_heads"].get<std::uint64_t>(),
                                      config["head_dim"].get<std::uint64_t>(),
                                      "attention query width");
        }
        if (config.contains("linear_num_key_heads") && config.contains("linear_key_head_dim")) {
            key_width = checked_mul(config["linear_num_key_heads"].get<std::uint64_t>(),
                                    config["linear_key_head_dim"].get<std::uint64_t>(),
                                    "GDN key width");
        }
        if (config.contains("linear_num_value_heads") &&
            config.contains("linear_value_head_dim")) {
            value_width = checked_mul(config["linear_num_value_heads"].get<std::uint64_t>(),
                                      config["linear_value_head_dim"].get<std::uint64_t>(),
                                      "GDN value width");
        }
        auto emit = [&](const std::string& name, ObjectHandle handle, std::uint64_t begin,
                        std::uint64_t end) {
            Binding binding;
            binding.parts.push_back(Part{handle, begin, end});
            binding.elements = end - begin;
            directory.bindings.emplace(name, std::move(binding));
        };
        for (std::size_t i = 0; i < directory.objects.size(); ++i) {
            const auto handle = ObjectHandle{i};
            const auto* tensor = std::get_if<TensorObject>(&directory.object(handle));
            if (!tensor) { continue; }  // resource objects carry no parameter bindings
            const auto names = logical_names(tensor->id, mtp_block);
            if (names.empty()) { continue; }
            const auto elements = shape_elements(tensor->shape);
            if (names.size() == 1) {
                Binding binding;
                binding.whole_object = true;
                binding.parts.push_back(Part{handle, 0, elements});
                binding.elements = elements;
                directory.bindings.emplace(names.front(), std::move(binding));
                continue;
            }
            const auto h = tensor->shape.size() >= 2 ? tensor->shape[1] : 0;
            if (names.front().ends_with("gdn/query")) {
                // GDN combined qkv: row order query, key, value.
                if (!key_width || !value_width) {
                    throw ArtifactError(tensor->id + ": GDN qkv rows need the GDN widths");
                }
                const auto rows =
                    checked_add(checked_mul(2, key_width, "GDN qkv rows"), value_width,
                                "GDN qkv rows");
                if (tensor->shape[0] != rows) {
                    throw ArtifactError(
                        tensor->id + ": GDN qkv rows disagree with the GDN widths");
                }
                const auto b0 = std::uint64_t{0};
                const auto b1 = checked_mul(key_width, h, "GDN qkv key offset");
                const auto b2 = checked_mul(checked_mul(2, key_width, "GDN qkv rows"), h,
                                             "GDN qkv value offset");
                const auto b3 = checked_mul(rows, h, "GDN qkv elements");
                if (b3 != elements) {
                    throw ArtifactError(
                        tensor->id + ": GDN qkv parts exceed the object elements");
                }
                emit(names[0], handle, b0, b1);
                emit(names[1], handle, b1, b2);
                emit(names[2], handle, b2, b3);
            } else if (names.front().ends_with("attention/query")) {
                // Full-attention combined q: row order query, gate.
                if (!query_width) {
                    throw ArtifactError(tensor->id + ": attention q rows need the query width");
                }
                const auto rows = checked_mul(2, query_width, "attention q rows");
                if (tensor->shape[0] != rows) {
                    throw ArtifactError(
                        tensor->id + ": attention q rows disagree with the query width");
                }
                const auto b0 = std::uint64_t{0};
                const auto b1 = checked_mul(query_width, h, "attention q offset");
                const auto b2 = checked_mul(rows, h, "attention q elements");
                if (b2 != elements) {
                    throw ArtifactError(
                        tensor->id + ": attention q parts exceed the object elements");
                }
                emit(names[0], handle, b0, b1);
                emit(names[1], handle, b1, b2);
            } else {
                throw ArtifactError("unexpected combined tensor " + tensor->id);
            }
        }
        if (mtp_block != std::numeric_limits<std::size_t>::max()) {
            // Alias the proposal (draft) head to the shared LM-head object. The materializer
            // uploads each object once (keyed by index), so this adds no device memory.
            const auto head = directory.bindings.find("text/output_head");
            if (head != directory.bindings.end()) {
                directory.bindings.emplace("proposal/head", head->second);
            }
        }
    }

    // The GGUF artifact carries no uses section, but the loader resolves every parameter
    // input against the directory (Binder::use) and requires each Use to carry an
    // activation policy. Synthesize the complete qwen3_5 uses set: per-block mixer and FFN
    // edges for every text layer and the MTP block, plus the output-head edges (the MTP
    // head aliases the text head parameter, see load/mtp.cpp). All heads consume 16-bit
    // activations, so every policy is A16Only (the .ninfer recipe sets the same).
    void build_uses() {
        auto add = [&](const std::string& parameter, const std::string& input) {
            Use use;
            use.parameter          = parameter;
            use.input              = input;
            use.activation_policy  = ActivationPolicy::A16Only;
            directory.uses.emplace(std::pair{use.parameter, use.input}, std::move(use));
        };
        auto add_block = [&](const std::string& prefix, bool full_attention) {
            if (full_attention) {
                add(prefix + "attention/query", prefix + "mixer_input");
                add(prefix + "attention/key", prefix + "mixer_input");
                add(prefix + "attention/gate", prefix + "mixer_input");
                add(prefix + "attention/value", prefix + "mixer_input");
                add(prefix + "attention/output", prefix + "attention/gated_output");
            } else {
                add(prefix + "gdn/query", prefix + "mixer_input");
                add(prefix + "gdn/key", prefix + "mixer_input");
                add(prefix + "gdn/value", prefix + "mixer_input");
                add(prefix + "gdn/z", prefix + "mixer_input");
                add(prefix + "gdn/a_projection", prefix + "mixer_input");
                add(prefix + "gdn/b_projection", prefix + "mixer_input");
                add(prefix + "gdn/output", prefix + "gdn/gated_output");
            }
            add(prefix + "mlp/gate", prefix + "ffn_input");
            add(prefix + "mlp/up", prefix + "ffn_input");
            add(prefix + "mlp/down", prefix + "mlp/product");
        };
        const auto block_count = meta_u64(model, "qwen35.block_count");
        const auto nextn = meta_optional(model, "qwen35.nextn_predict_layers")
                               ? meta_u64(model, "qwen35.nextn_predict_layers")
                               : std::uint64_t{0};
        const auto num_layers = block_count - nextn;
        const auto interval = meta_u64(model, "qwen35.full_attention_interval");
        for (std::uint64_t i = 0; i < num_layers; ++i) {
            add_block("text/layers/" + std::to_string(i) + "/", (i + 1) % interval == 0);
        }
        add("text/output_head", "text/final_hidden");
        if (mtp_block != std::numeric_limits<std::size_t>::max()) {
            add("text/output_head", "mtp/final_hidden");
            add("proposal/head", "mtp/final_hidden");
            add("mtp/input_projection", "mtp/stem_input");
            add_block("mtp/layers/0/", true);
        }
    }

    // GGUF artifacts carry no tokenizer: they are token-input-only (prepare_tokens /
    // score_tokens / generate with token ids; text prompts are unavailable without a
    // tokenizer sidecar). Synthesize empty ResourceObjects for the four text roles the
    // binder requests so bind_resources reads empty strings without throwing.
    void build_resources() {
        // The GGUF carries its tokenizer in the metadata; render it as the HF resource strings the
        // frontend reads. A file without an embedded tokenizer keeps the empty token-only
        // resources. strata's parsed metadata keeps only a 64-item sample per array, so the
        // vocabulary and merges are re-read from the mmapped header.
        std::string tokenizer_text, tokenizer_config_text, generation_text, chat_template_text;
        if (shard_bases.front() != nullptr) {
            const auto tokenizer =
                read_gguf_tokenizer(shard_bases.front(), model.shard(0).file_size());
            if (!tokenizer.tokens.empty()) {
                tokenizer_text        = tokenizer_json(tokenizer).dump();
                tokenizer_config_text = tokenizer_config_json(tokenizer).dump();
                generation_text       = generation_config_json(tokenizer).dump();
                if (tokenizer.has_chat_template) { chat_template_text = tokenizer.chat_template; }
            }
        }
        const auto emit = [&](const char* role, const std::string& text) {
            ResourceObject resource;
            resource.id       = std::string("text/") + role;
            resource.encoding = kRawBytesEncoding;
            resource.offset   = resources_payload_.size();
            resource.bytes    = text.size();
            resources_payload_.insert(resources_payload_.end(),
                                      reinterpret_cast<const std::byte*>(text.data()),
                                      reinterpret_cast<const std::byte*>(text.data()) +
                                          text.size());
            const auto handle = ObjectHandle{directory.objects.size()};
            directory.objects.push_back(std::move(resource));
            directory.object_index.emplace(std::string("text/") + role, handle);
            directory.components.at("text").resources.emplace(role, handle);
        };
        emit("tokenizer.json", tokenizer_text);
        emit("tokenizer_config.json", tokenizer_config_text);
        emit("chat_template.jinja", chat_template_text);
        emit("generation_config.json", generation_text);
    }
};

GgufReader::GgufReader(const std::filesystem::path& path) : Reader() {
    try {
        impl_ = std::make_unique<Impl>(path);
    } catch (const ArtifactError&) {
        throw;
    } catch (const std::exception& error) {
        throw ArtifactError(path.string() + ": " + error.what());
    }
}

GgufReader::GgufReader(GgufReader&&) noexcept = default;
GgufReader& GgufReader::operator=(GgufReader&&) noexcept = default;

// Defined where GgufReader::Impl is complete (the base Reader destructor is defined the
// same way in reader.cpp).
GgufReader::~GgufReader() = default;

const Directory& GgufReader::directory() const noexcept { return impl_->directory; }

const ArtifactId& GgufReader::artifact_id() const noexcept {
    // GGUF files carry no NInfer artifact id; the id stays zeroed.
    return impl_->id;
}

std::uint64_t GgufReader::file_bytes() const noexcept { return impl_->file_bytes; }

ObjectHandle GgufReader::find(std::string_view id) const {
    const auto found = directory().object_index.find(std::string(id));
    if (found == directory().object_index.end()) {
        throw ArtifactError("missing object " + std::string(id));
    }
    return found->second;
}

const WeightGeometry& GgufReader::geometry(ObjectHandle handle) const {
    const auto& object = directory().tensor(handle);
    auto& slot         = impl_->geometries[handle.index];
    if (!slot) {
        WeightGeometry geometry;
        geometry.shape     = object.shape;
        geometry.elements  = shape_elements(object.shape);
        geometry.alignment = 256;
        if (impl_->cast_offsets_.contains(handle.index)) {
            // Owned F32->BF16 cast (see build_objects): the bytes are reader-owned, not a
            // file-backed GgufNative reference.
            geometry.format        = QType::BF16;
            geometry.layout        = QuantLayout::Contiguous;
            geometry.bytes         = object.bytes;
            geometry.owned_payload = true;
        } else {
            std::size_t shard = 0;
            const auto* tensor = impl_->model.find(object.id, &shard);
            const auto bytes   = strata::tensor_payload_bytes(*tensor);
            if (!bytes) { throw ArtifactError(object.id + ": tensor payload size is unknown"); }
            geometry.format    = gguf_qtype(tensor->type);
            geometry.bytes     = bytes;
            if (tensor->type == 0 || tensor->type == 30) {
                // Direct GGUF float (exact FP32 or BF16): contiguous, like the cast path above.
                geometry.layout = QuantLayout::Contiguous;
            } else {
                geometry.layout    = QuantLayout::GgufNative;
                geometry.ggml_type = static_cast<std::uint16_t>(tensor->type);
                // GgufNative is a matrix: the K column count is the padded column extent, and the
                // per-row code byte count is the payload split across the N rows. native_weight()
                // reads both (dimension(padded_columns) and the row-plane offset), so they must be
                // set here — weight_geometry() is never used for this layout.
                if (object.shape.size() == 2 && object.shape[0]) {
                    geometry.padded_columns     = object.shape[1];
                    geometry.code_bytes_per_row = bytes / object.shape[0];
                }
            }
        }
        slot = std::move(geometry);
    }
    return *slot;
}

void GgufReader::validate_object(ObjectHandle handle) const {
    const auto& object = directory().object(handle);
    if (std::holds_alternative<ResourceObject>(object)) {
        // Synthetic empty token-only resource: nothing to validate.
        return;
    }
    const auto& tensor = directory().tensor(handle);
    std::size_t shard  = 0;
    const auto* info   = impl_->model.find(tensor.id, &shard);
    if (!info || !impl_->model.in_bounds(*info, shard)) {
        throw ArtifactError(tensor.id + ": tensor lies outside its shard data section");
    }
    (void)geometry(handle);
}

std::vector<ReadSegment> GgufReader::segments(std::uint64_t offset, std::uint64_t bytes) const {
    const auto total = impl_->directory.payload_bytes;
    if (offset > total || bytes > total - offset) {
        throw ArtifactError("read range exceeds logical payload");
    }
    std::vector<ReadSegment> result;
    if (!bytes) { return result; }
    std::uint64_t copied = 0;
    for (std::size_t s = 0; copied < bytes && s < impl_->payload_sizes.size(); ++s) {
        const auto begin     = impl_->logical_begins[s];
        const auto shard_end = checked_add(begin, impl_->payload_sizes[s], "shard payload end");
        const auto read_begin = offset + copied;
        if (shard_end <= read_begin) { continue; }
        if (begin >= offset + bytes) { break; }
        const auto local = read_begin > begin ? read_begin - begin : 0;
        const auto count = std::min(bytes - copied, shard_end - (read_begin > begin ? read_begin
                                                                                   : begin));
        result.push_back(
            {s, checked_add(impl_->data_starts[s], local, "file offset"), copied, count});
        copied += count;
    }
    return result;
}

void GgufReader::read_into(std::uint64_t offset, std::span<std::byte> destination) const {
    for (const auto& segment : segments(offset, destination.size())) {
        const auto* base = impl_->shard_bases[segment.file_index];
        if (!base) {
            throw ArtifactError("shard " + std::to_string(segment.file_index) +
                                " has no readable data section");
        }
        std::memcpy(destination.subspan(static_cast<std::size_t>(segment.destination_offset),
                                        static_cast<std::size_t>(segment.bytes))
                        .data(),
                    base + segment.file_offset, segment.bytes);
    }
}

std::vector<std::byte> GgufReader::read_range(std::uint64_t offset, std::uint64_t bytes) const {
    if (bytes > std::numeric_limits<std::size_t>::max()) {
        throw ArtifactError("Host read size exceeds size_t");
    }
    std::vector<std::byte> result(static_cast<std::size_t>(bytes));
    read_into(offset, result);
    return result;
}

std::vector<std::byte> GgufReader::read_object(ObjectHandle handle) const {
    const auto& object = directory().object(handle);
    if (const auto* resource = std::get_if<ResourceObject>(&object)) {
        // Synthesized frontend resource (see build_resources); an absent tokenizer leaves it empty.
        return {impl_->resources_payload_.begin() + resource->offset,
                impl_->resources_payload_.begin() + resource->offset + resource->bytes};
    }
    if (const auto cast = impl_->cast_offsets_.find(handle.index);
        cast != impl_->cast_offsets_.end()) {
        // Owned F32->BF16 cast payload (see build_objects): serve the cast bytes, not the
        // F32 source tensor.
        const auto& tensor = directory().tensor(handle);
        const auto begin   = cast->second;
        return {impl_->cast_payload_.begin() + begin,
                impl_->cast_payload_.begin() + begin + tensor.bytes};
    }
    validate_object(handle);
    const auto& tensor = directory().tensor(handle);
    std::size_t shard  = 0;
    const auto* info   = impl_->model.find(tensor.id, &shard);
    const auto bytes   = strata::tensor_payload_bytes(*info);
    std::vector<std::byte> result(static_cast<std::size_t>(bytes));
    std::memcpy(result.data(), impl_->model.shard(shard).tensor_data(*info), bytes);
    return result;
}

std::size_t GgufReader::read_direct(std::size_t file_index, std::uint64_t file_offset,
                                    std::span<std::byte> destination) const {
    if (file_index >= impl_->model.size()) { throw ArtifactError("invalid shard index"); }
    const auto* base = impl_->shard_bases[file_index];
    if (!base) {
        throw ArtifactError("shard " + std::to_string(file_index) +
                            " has no readable data section");
    }
    const auto& shard = impl_->model.shard(file_index);
    if (file_offset > shard.file_size()) {
        throw ArtifactError("direct read offset exceeds the shard size");
    }
    const auto available = shard.file_size() - file_offset;
    const auto count     = std::min<std::uint64_t>(available, destination.size());
    std::memcpy(destination.data(), base + file_offset, count);
    return static_cast<std::size_t>(count);
}

} // namespace ninfer::artifact
