# HTTP serving

`build/apps/ninfer-serve` exposes OpenAI- and Anthropic-compatible HTTP endpoints over the
in-process multi-model router. The model list and every per-model engine preset come from a
serve-config JSON file (`--config`, required); the process starts **no-resident** and loads a
model on demand when the first request for it arrives, swapping the single GPU-resident model as
needed. CLI flags cover only the global server, memory, and ingress options.

## Start the server

See [CUDA synchronization](cli.md#cuda-synchronization) for the shared `NINFER_CUDA_SYNC` setting.

```bash
cat > serve-config.json <<'EOF'
{
  "models": {
    "qwen3.8-27b/nvfp4": {
      "artifact": "models/qwen3_8_27b_nvfp4.ninfer",
      "maxContext": 240000,
      "kvCapacity": 240000,
      "kvDtype": "fp8",
      "spec": "mtp",
      "draftTokens": 3,
      "lmHeadDraft": true,
      "vision": true
    }
  },
  "healthCheckTimeout": 300,
  "globalTTL": 300
}
EOF

./build/apps/ninfer-serve \
  --config serve-config.json \
  --host 127.0.0.1 \
  --port 8080 \
  --max-concurrency 2 \
  --device-state-slots 2 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

 The command uses Qwen3.8-27B NVFP4. Each request has a 240,000-token logical ceiling. A shared
 240,000-token Main Text KV pool serves admitted requests; either request may use the full capacity
 when running alone. Requests acquire KV pages as execution advances; if concurrent growth exhausts
 the pool, the scheduler can pause a request and restore it later.

With `C=2` and two extra Device slots, the process owns four Device StateImages. The default shared
pinned Host budget is 8 GiB plus eight model StateImages. It holds retained state, KV and pause
snapshots, including in-flight destinations; `--host-context-mib` sets an explicit total instead.

Other artifacts use the same shape with their own `models` entry (and their own engine presets).
For 35B-A3B DFlash, set the model's speculative fields to `"spec": "dflash"`,
`"draftTokens": 7`, `"lmHeadDraft": true` in its config entry. Qwen3.8-27B artifacts with
DFlash2 companion weights also support `"spec": "dflash2"` with `"draftTokens": 7`, leaving
`lmHeadDraft` optional. DFlash2 accepts draft counts 1..15 and supports the same sampling,
concurrency, prefix reuse, and image/video request surfaces. It may remain combined with
`"vision": true`.

When `--model-id` is omitted, the server advertises and accepts the artifact's `metadata.name`,
falling back to its architecture name when no name is stored. An explicit `--model-id` is a public
HTTP alias override and does not select or alter model execution.

Vision is disabled by default: its weights and Vision-specific unified-workspace extent are not
allocated, and media requests and token-count requests fail with HTTP 400 `vision_disabled`. Set
the model's `"vision": true` in the serve config when the model must accept image or video input.
Speculative residency is likewise frozen by the model's `spec` (`mtp|dflash|dflash2`) and
`draftTokens` fields; omitting `spec` loads no speculative backend. `lmHeadDraft` additionally
loads the optimized proposal head. DFlash on 35B-A3B and DFlash2 on Qwen3.8-27B can be combined
with `"vision": true`; each accelerates generated-text decode after multimodal prefill, while
Vision encode and prefill remain outside speculative acceleration. A later request cannot enable
a capability omitted at load.

## Endpoints

| Method and path | Behavior |
|---|---|
| `GET /health` | Engine readiness |
| `GET /metrics` | Prometheus counters, gauges and latency histograms |
| `GET /v1/models` | configured OpenAI model alias and effective `max_model_len` |
| `GET /v1/models/{id}` | lookup of the configured alias and effective `max_model_len` |
| `POST /v1/chat/completions` | OpenAI-style chat generation |
| `POST /v1/responses` | OpenAI Responses Core generation, state, typed Items, and SSE |
| `POST /v1/responses/input_tokens` | Responses prompt-token count without generation |
| `GET /v1/responses/{id}` | retrieve a locally stored terminal Response |
| `DELETE /v1/responses/{id}` | delete a locally stored Response |
| `GET /v1/responses/{id}/input_items` | list that Response's normalized input Items |
| `POST /v1/messages` | Anthropic-style message generation |
| `POST /v1/messages/count_tokens` | checkpoint-native expanded input-token count |
| `POST /v1/decisions` | Decision scoring over a state + questions, on the loaded model |
| `POST /v1/systemone` | Alias of `POST /v1/decisions` (identical handler and contract) |
| `POST /models/load` | llama.cpp compatibility: load/swap the resident model |
| `POST /models/unload` | llama.cpp compatibility: force the no-resident state |
| `GET /models/sse` | llama.cpp compatibility: live model-status SSE feed |
| `GET /slots` | llama.cpp compatibility: live concurrency slot state |
| `GET /tools` | llama.cpp compatibility: list registered server-side tools |
| `POST /tools` | llama.cpp compatibility: execute a server-side tool |
| `POST /v1/streams/lookup` | llama.cpp compatibility: list active conversation streams |
| `GET /v1/stream/{id}` | llama.cpp compatibility: replay + live-tail a registered stream |
| `DELETE /v1/stream/{id}` | llama.cpp compatibility: cancel + remove a registered stream |
| `POST /v1/chat/completions/control` | llama.cpp compatibility: cancel an in-flight generation |

`GET /health` is a pure liveness probe: while the server process is up it always returns HTTP 200
with `{"status":"ok"}`, independent of model load state (a startup /health with nothing loaded is
still 200). Per-model readiness is surfaced via the `GET /v1/models` `status` field
(`loaded`/`unloaded`), not via /health. The endpoint remains unauthenticated.

Every OpenAI-compatible response carries a unique `x-request-id` header, including streaming and
error responses. Anthropic endpoints use their separate `request-id` contract.

All three generation SSE endpoints emit the standard `: keep-alive` comment after five seconds
without a protocol event. The comment is transport-only: SSE clients ignore it, and it does not
change generated text, event ordering, usage, stored Responses, or request logs. On Linux, accepted
connections also use TCP keepalive and a 15-second `TCP_USER_TIMEOUT`; together with the heartbeat,
a dead or unacknowledging peer is normally cancelled within about 20 seconds, including while the
request is waiting or prefilling. A peer whose TCP stack remains connected and acknowledges data
cannot be distinguished from a reading application; proxies must close their upstream NInfer
connection when the downstream client disappears.

## OpenAI Chat Completions

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "What is speculative decoding?"}
    ],
    "max_tokens": 128
  }'
```

The endpoint supports:

- `system`, `developer`, `user`, `assistant`, and `tool` history, plus legacy `function` history;
- string content and ordered text/refusal parts; adjacent parts are preserved without inserted
  separators, and empty wire content remains an empty turn;
- User `image_url` parts, tool-result `image_url` parts used by compatible clients, and the User
  `video_url` extension using HTTP(S) or data URIs; image detail is omitted or `auto`;
- nonnegative `max_completion_tokens` and the legacy `max_tokens` spelling; a non-positive or
  omitted output budget resolves to the server `defaultMaxTokens` (clamped to the engine's context
  capacity);
- `temperature`, `top_p`, presence/frequency penalties, and nonnegative integer `seed`;
- the compatible `top_k` (`0..20`) and `min_p` (`0..1`) sampler extensions;
- up to four non-empty stop strings, applied to both reasoning and answer output;
- `n:1`, text-only `modalities`, and `response_format: {"type":"text"}`;
- non-streaming responses and server-sent event streams;
- `stream_options.include_usage`;
- llama.cpp-compatible terminal `timings`, plus opt-in `timings_per_token` and
  streaming `return_progress` observations;
- function tools, optional `strict:true` argument schemas, `tool_choice` `auto`/`none`/`required`,
  named selection, `allowed_tools`, and `parallel_tool_calls`; assistant tool-call history,
  tool-result messages, and legacy function-call history;
- the top-level `reasoning_effort` field;
- `enable_thinking` and `preserve_thinking`, either at top level or in
  `chat_template_kwargs`;
- Assistant `reasoning_content` and `reasoning` history aliases.

Options whose observable behavior the Engine cannot provide are rejected when they request that
behavior. This includes nonzero `logit_bias`, requested log probabilities,
audio/file input or audio output, explicit low/high image detail, web search,
moderation, low/high verbosity, stored Chat Completions, and non-empty legacy `functions`.
Each capability rejection identifies the affected field and the guarantee NInfer cannot provide.

JSON mode and JSON Schema use the standard protocol fields:

| Endpoint | Field |
|---|---|
| Chat Completions | `response_format: {"type":"json_object"}` or `{"type":"json_schema","json_schema":{"name":"answer","schema":{...},"strict":true}}` |
| Responses | `text.format: {"type":"json_object"}` or `{"type":"json_schema","name":"answer","schema":{...},"strict":true}` |
| Anthropic Messages | `output_config.format: {"type":"json_schema","schema":{...}}` |

`json_object` requires an object root. Schema mode follows the supplied root type and enforces the
[supported assertions](maintainer/constrained-decoding.md#42-json-与-schema-的执行合同), including when
`strict` is omitted or false. Unsupported assertions return HTTP 400 before generation. OpenAI
errors distinguish `invalid_json_schema`, `unsupported_json_schema` and `unsatisfiable_json_schema`;
`param` identifies the request field followed by the schema JSON Pointer. Anthropic uses its
`invalid_request_error` envelope with the schema location in the message. Responses echoes the
selected `text.format` in aggregate responses and SSE response objects.

JSON output uses compact separators and declared property order. State the desired content in the
prompt; the schema is not inserted into it. Only one output constraint may be supplied.

Schemas support positional arrays (`prefixItems` plus tail `items`) and inclusive/exclusive
`number` ranges. Bounded numbers use exact int64 integers or finite binary64-compatible decimal
and scientific notation with up to 17 significant digits. Bounds must retain their value when the
schema is parsed; numbers requiring greater precision receive `unsupported_json_schema`.
These capabilities also apply to strict tool parameters.

GBNF, choice and regex are available through the NInfer extension `structured_outputs`
on Chat Completions, Responses and Anthropic Messages. Supply exactly one member:

```json
{"structured_outputs": {"grammar": "root ::= \"yes\" | \"no\""}}
```

```json
{"structured_outputs": {"choice": ["positive", "neutral", "negative"]}}
```

```json
{"structured_outputs": {"regex": "(BUG|TASK)-[0-9]{4}"}}
```

Choice returns one literal string, preserving case and whitespace. The list must be nonempty;
duplicate entries have no extra weight, and an empty-string entry permits empty content.
Regex matches the complete content. It supports character classes, groups, alternatives and
repetition; `.` excludes line terminators, `\d`/`\w` use ASCII ranges, and `\s` includes Unicode
whitespace. Empty regex permits only empty content. Anchors are supported at the ends of top-level
alternatives. Lookaround, backreferences, word boundaries, Unicode properties, flags and unknown
escapes return HTTP 400. See the [language contract](maintainer/constrained-decoding.md#41-gbnf--regex--choice).
Invalid choices and regexes use `invalid_choice` and `invalid_regex`, with the request field in `param`.

These constraints apply to answer content; thinking is separate. GBNF supports recursive rules,
Unicode and repetition. All modes support streaming and all speculative backends. For assistant
continuation, the grammar covers the existing assistant content plus the generated suffix. Completion uses the model's EOS tokens;
output limits and cancellation can produce an incomplete answer. JSON modes can be combined with
active tools; GBNF, choice and regex require no active tools or `tool_choice:"none"`.
Output constraints reject custom stops. OpenAI errors use
`invalid_grammar` for invalid grammars and `constraint_dead_end` for a reachable prefix without a
legal next token. Anthropic reports these through its `invalid_request_error` envelope.
The `grammar` and `guided_*` aliases are not accepted.

Constrained responses include a NInfer `constraint` observation. `branch` is `undecided`, `content`,
or `tools`; `complete` means the committed language can end, and `terminated` means it accepted EOS.
A complete JSON value can therefore have `complete:true`, `terminated:false` and a length finish
reason. The observation also includes `cache` (`hit`, `built`, `waited`), `mask_positions`,
`mask_upload_bytes`, and `timings_seconds` for preparation, CPU mask work and matcher work.
These times are parts of existing request time and can overlap GPU execution.
Streaming sends the observation once: the Chat finish/usage chunk, the Responses terminal response
object, or Anthropic `message_delta`. Unconstrained responses omit it.

Semantically neutral fields do not make an otherwise executable request fail. All-zero
`logit_bias`, `logprobs:false`, `top_logprobs:0`, `verbosity:"medium"`, empty legacy tool controls,
text-only `audio` configuration, and `prediction` are accepted without changing Engine execution.
Metadata, user/safety identifiers, service-tier and prompt-cache hints are likewise advisory.
Unknown top-level fields are ignored.

A string `name` on a `tool` message is accepted as an ignored, output-neutral compatibility
extension for clients that mirror the function name onto tool results. It does not participate in
tool identity, prompt rendering, or output. Non-string values are malformed; non-empty names on
other message roles remain unsupported because they carry participant identity that the loaded chat
template cannot represent.

For commonly generated OpenAI-compatible payloads, `repetition_penalty` is accepted only at its
neutral value `1`, and `mm_processor_kwargs` when empty or containing only null values. String-form
image/video URLs are also accepted.

Malformed protocol values return field-specific HTTP 400 errors. Invalid media sources, bytes, or
decoded content use `invalid_media`; remote fetch and timeout failures retain their dedicated
server-error codes. Failures in the normalized prompt contract use `invalid_prompt`; typed capacity
and availability failures retain their dedicated codes. Internal invariant failures are not
relabeled as client input errors.

The request `model` must equal the public model ID: the artifact `metadata.name` by default
(falling back to its architecture name when absent), or the explicit `--model-id` override. A
non-string `model` value is rejected with HTTP 400 code `invalid_request_error` (param `model`);
an empty string or `null` model is not rejected and resolves to the default. Reasoning is returned
separately as `reasoning_content`; answer text remains in `content`.

For non-strict tools, a direct top-level tool-parameter
`type`, or an `anyOf`/`oneOf` composed entirely of explicit primitive types, guides conversion of
Qwen's untyped parameter text. It does not decide whether structurally complete markup is a tool
call. String-admitting values remain strings, including the empty string; the only
exception is a tool-result message's `tool_call_id`, which must be non-empty. An empty block
for a declared non-string parameter is omitted. Admitted JSON values retain their JSON type;
case-insensitive boolean text is normalized to `true` or `false`. A nonempty schema mismatch remains
a structured call: valid JSON retains its represented type and other text becomes a JSON string so
the tool consumer can report the validation error and continue the agent loop. Schemas without a
supported explicit type retain untyped inference. NInfer does not apply defaults, enforce required
properties, or perform recursive JSON Schema validation on this route.

On the unconstrained route, string parameters preserve function/tool-call markers and balanced nested
`<parameter=...>...</parameter>` text as value bytes. The Qwen wire format has no delimiter escape,
so an unmatched nested parameter opener or a standalone `</parameter>` cannot be represented
unambiguously; either causes the complete tool-call region to fall back to ordinary content.

### Tool constraints

The three protocols share one constrained tool implementation:

| Choice | Generated calls |
|---|---|
| `auto` | Text or calls; zero to many |
| `none` | No tool calls; declarations remain in the prompt |
| OpenAI `required` / Anthropic `any` | One or more calls |
| OpenAI named function | Exactly one call to that function |
| Anthropic named `tool` | One or more calls to that tool |
| OpenAI `parallel_tool_calls:false` / Anthropic `disable_parallel_tool_use:true` | At most one call; exactly one when a call is required |

OpenAI `allowed_tools` supports `auto` and `required`. Selection changes generation permissions,
while all declarations retain their original order in the prompt. Requests with tools enable
**basic structural constraints by default**, including ordinary `auto` calls without `strict`.
The model can answer normally or start a tool call; a call must use the model's tool framing and a
declared function name.

Non-strict parameter names and order remain open. Their schema supplies the existing value
normalization hints; it is not compiled as a strict constraint. Open objects, root unions, and
unsupported schema assertions therefore remain usable. Repeated parameter names use the last
value, retaining the first key position; the published JSON object contains each key once.
`strict:true` additionally enforces the parameter contract below.

Top-level `tool_constraints:"auto"` opts into request-driven constraints: ordinary non-strict
`tool_choice:"auto"` then uses free generation. Strict tools, selection/count restrictions, and
`tool_choice:"none"` still enforce their requirements. `tool_constraints:"basic"` is the default.

With JSON object/schema output, `auto` permits either a JSON answer or a complete tool-call sequence.
Required/named choices permit calls for that turn; after supplying the tool result, use `auto` for
the final JSON answer. `none` permits only JSON. The JSON schema is validated on every turn.
This combination enforces tool framing even with `tool_constraints:"auto"`; `strict` continues to
control argument-value validation. Tool markers inside JSON strings remain ordinary string data.

A function's `strict:true` also constrains its argument values against its schema. Its parameter
root must reduce to a `type:"object"` schema with `additionalProperties:false`, including supported
`allOf` and local-reference combinations.
Properties are emitted in declaration order; optional properties may be omitted. Root
const/enum/unions are not supported. Values use the supported JSON Schema subset described above.
Top-level pure string parameters use raw text and preserve whitespace. Other values use JSON;
a top-level string/non-string union (such as string/null) is rejected because the Qwen parameter
format cannot distinguish those branches. Such unions inside JSON objects or arrays are supported.
Raw values cannot contain the delimiter `\n</parameter>`; unsatisfiable required values are rejected.
Integer arguments use signed 64-bit values; number arguments use finite binary64-compatible
representations. Unsupported schemas fail with HTTP 400 before generation.

For `auto` without JSON output, text can precede the first call. Required/named choices start directly with calls
(after thinking, if enabled). Once a constrained call starts, the suffix consists of complete calls
and model EOS. Active tool constraints require model EOS and reject custom stop strings.
For ordinary non-strict auto calls that need custom stops, select `tool_constraints:"auto"`.
Token limits and cancellation can still stop generation: only completed calls are published.
A later call truncated by the token limit keeps `length`/`max_tokens`/Responses `incomplete` as the
terminal status. Streaming publishes each completed call in the terminal event sequence;
arguments are not streamed incrementally.
Assistant continuation may finish a partial call; a prefix containing a completed call is rejected.

Messages enter the selected template in their input order. The maintained Qwen templates keep
system/developer messages at their original positions.

Prompt-bearing JSON objects retain their received member order through request parsing and prompt
rendering, including tool schemas and historical tool inputs. Canonical model-origin tool arguments
retain that member order in aggregate and streaming responses, so an unmodified replay reconstructs
the same ordered tool call. NInfer does not canonicalize semantically equivalent JSON: if a client
reorders members, inserts defaults, or otherwise rewrites a tool object, the changed rendered input
does not match the model-held endpoint and can reuse only an earlier exact checkpoint.
Generated token segmentation can also differ from re-encoding the same text, limiting prefix reuse.

`--chat-template FILE` selects a local Jinja template; by default, the server uses the template
stored in the artifact. See the [CLI guide](cli.md#text-input) for an example.

Control-token spellings quoted in message content, tool data or ordinary template kwargs are
encoded as text. Media placeholders come from the template and bind to actual image/video inputs.

`chat_template_kwargs` passes a JSON object to the template in Chat Completions, Responses and
Anthropic Messages. Values duplicated in typed request fields must agree. Null standard options
mean unspecified; other null values remain `none`. Messages, tools, generation mode and tokenizer
special tokens cannot be overridden through kwargs.

`--default-thinking-budget N` sets a positive default thinking-token cap for requests that start
in thinking mode. Non-thinking requests receive no cap. It may coexist with `--no-thinking`
because requests can explicitly enable thinking. Anthropic
`thinking:{"type":"enabled","budget_tokens":N}` overrides this default for that request.

Add `--default-thinking-budget 512` to the startup command to cap model-origin thinking at 512
tokens for every thinking-enabled request.

At the cap boundary, Engine first honors a natural `</think>`, stop condition, cancellation, or
total output/context limit. If thinking remains open, it commits Qwen's canonical early-close
guidance and close marker to the same model sequence without sampling, streams the guidance as a
reasoning delta, and continues normal content or tool-call generation. Inserted tokens count in
completion usage and the request's `max_tokens`/`max_output_tokens` budget. If the effective output
capacity extends past the cap but cannot fit the complete tokenizer-derived control suffix plus one
post-close model token, preparation is rejected with HTTP 400 code
`thinking_budget_capacity_insufficient` rather than partially inserting control. The server does
not promise that the model will emit nonempty content or a tool call after the marker.

For Chat Completions, `reasoning_effort: "none"` requests disabled thinking. The selected template
interprets the other standard values (`minimal`, `low`, `medium`, `high`, `xhigh`, `max`).
Conflicting explicit `enable_thinking` and effort values return `conflicting_template_option`.

`preserve_thinking` controls reasoning retention according to the selected template. Request
options override server defaults set with `--no-thinking` and `--preserve-thinking`. Unspecified
thinking, effort and preservation options use the template's defaults.

Streaming begins with an assistant-role chunk, sends separate reasoning and content deltas, then a
finish-reason chunk and `[DONE]`. When `stream_options.include_usage` is true, a final empty
`choices` chunk contains completed usage. Aggregate and streamed usage include cached prompt tokens
and reasoning-token details; choices carry `logprobs: null` when log probabilities were not
requested, and aggregate assistant messages carry `refusal: null` because refusal output is not
supported.

A mid-stream failure (a generation, admission, or render error) does not abort with an HTTP
error status — the stream is already open. Instead the endpoint emits an in-stream error frame:
a `data:` line whose JSON body is the same error object a non-streaming failure returns
(`{"error":{"message":…,"type":…,"param":…,"code":…}}`), then closes the stream. The error
frame is the last frame: no finish-reason chunk and no `[DONE]` follow it. Clients reading a
chat stream must handle a `data:` frame whose body is an error object (rather than a chunk)
appearing mid-stream, and treat the stream as terminated when it closes without `[DONE]`.

### llama.cpp-compatible request observations

Every successful Chat Completions response includes a top-level `timings` object. This is a
llama.cpp-compatible response extension, not an OpenAI field. In a stream it is attached to the
last JSON chunk before `[DONE]`: the empty `choices` usage chunk when
`stream_options.include_usage` is true, otherwise the finish-reason chunk.

```json
{
  "timings": {
    "cache_n": 4096,
    "prompt_n": 4096,
    "prompt_ms": 83.0,
    "prompt_per_token_ms": 0.020263671875,
    "prompt_per_second": 49349.39759036145,
    "predicted_n": 129,
    "predicted_ms": 1140.0,
    "predicted_per_token_ms": 8.90625,
    "predicted_per_second": 112.28070175438596
  }
}
```

`cache_n` is the exact Engine-proven reused prompt prefix and `prompt_n` is the remaining prompt
suffix, so `cache_n + prompt_n` equals `usage.prompt_tokens`. Prompt time starts when admission
commits that exact reuse choice and ends when the first output token is committed. Generation time
starts at that first token and ends at the last committed output token. Accordingly, generation
speed uses `max(predicted_n - 1, 0)` token intervals; the first token belongs to prompt latency and
is not counted again as a decode interval. Zero-token, one-token, zero-duration, and exact-cache-hit
cases report finite zero rates rather than `NaN` or infinity. Speculative requests additionally
include terminal `draft_n` and `draft_n_accepted` when draft work occurred.

Set top-level `timings_per_token: true` on a streaming request to attach the latest cumulative
timing snapshot to each visible reasoning or content chunk. This does not enable terminal timings,
which are always present. A model commit that is temporarily hidden by UTF-8, stop-string,
reasoning, or tool-call buffering still advances the cumulative token count; the next visible chunk
observes that committed frontier. The option increases response serialization and transport volume
and is off by default.

Set top-level `return_progress: true` together with `stream: true` to receive prompt-processing
chunks:

```json
{
  "prompt_progress": {
    "total": 8192,
    "cache": 4096,
    "processed": 6144,
    "time_ms": 41
  }
}
```

The initial event has `processed == cache`. Later cumulative events are published only after the
corresponding prefill unit commits, may be coalesced when the consumer is slower than prefill, and
never move backwards. The final event has `processed == total` and precedes the first output delta.
For an exact full-prefix hit, the initial event already has `cache == processed == total` and no
synthetic prompt work is reported. `time_ms` is elapsed wall time since committed admission;
clients may calculate actual suffix progress as `(processed-cache)/(total-cache)` when the
denominator is nonzero.

### Multimodal request

Start the model with its `"vision": true` serve-config field before sending media:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": "Describe this image."}
      ]
    }],
    "max_tokens": 128
  }'
```

OpenAI image and video sources may be HTTP(S) URLs or base64 data URLs.

Text and media requests use one complete-prompt context contract. After chat-template rendering and
media-token expansion, the result must fit the model's context ceiling (`maxContext` in the serve
config). The current Vision runtime also has a 32,768 merged-token envelope (131,072 raw patches);
the effective Vision limit is therefore `min(maxContext, 32768)`. There is no fixed image/video
item-count limit: item count is admitted through aggregate source-byte, decoded-pixel, raw-patch,
Vision-token, and live-memory budgets.

Media cache misses run as independent decode → resize → BF16-pack tasks on a bounded host worker
pool. Prepared payloads are keyed by SHA-256 of the acquired bytes plus modality, so repeated media
in later requests reuses the exact immutable BF16 patch input; concurrent identical misses use one
single-flight build. `--media-cache-mib` bounds LRU-retained payloads, while
`--media-live-mib` bounds every cache-, request-, or runtime-referenced payload. Cache eviction does
not invalidate a request reference, and live bytes are returned only when the final reference is
released. A request-level preparation gate derived from the live limit prevents concurrent partial
builds from deadlocking the memory account.

An expanded prompt beyond the model's context ceiling (`maxContext`) returns HTTP 400
`context_length_exceeded`, including the prepared token count and configured context ceiling. A media preprocessing resource rejection
returns HTTP 400 `media_budget_exceeded`. HTTP 413 `request_too_large` is reserved for a raw request
body that exceeds `--max-request-mib` before JSON parsing; it is not used for model-context or media
resource errors.

## OpenAI prompt caching

Chat Completions and Responses translate OpenAI cache hints into optional shared-prefix write
candidates:

- omitted `prompt_cache_options` creates a default implicit candidate at the end of the latest
  cacheable content part;
- `mode:"implicit"` requests the same automatic candidate explicitly;
- `mode:"explicit"` disables that implicit write for the request;
- `prompt_cache_breakpoint:{"mode":"explicit"}` on supported content creates an explicit
  candidate.

The automatic candidate precedes the message closing tokens, allowing reuse when the same message
body grows and its previous tokens remain an exact prefix. An explicit marker keeps its requested
location. Responses applies this policy after expanding stored history.

One request carries at most four writes. Explicit markers take precedence: four explicit candidates
leave no extra slot for an automatic candidate. If more explicit markers appear in the history, the
latest four remain write candidates. Exact reads of already-published prefixes do not require the
request to repeat a marker.

These fields are optimization hints. A legal boundary that cannot be represented as an exact
rendered-token frontier is ignored without changing prompt content. `prompt_cache_key` is not an
Engine session key or prefix identity. Valid TTL/retention values are accepted, but NInfer does not
promise their wall-clock residency; physical retention follows the resource scheduler.

## OpenAI Responses Core

NInfer implements the typed-Item and semantic-event core of the OpenAI
[Responses API](https://developers.openai.com/api/reference/resources/responses/overview). All
supported model instances use this same adapter and Engine route. It is intentionally not
advertised as full parity with OpenAI-hosted tools, durable cloud storage, background jobs,
Conversations, or compaction.

### Create a Response

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "instructions": "Answer concisely.",
    "input": "What is speculative decoding?",
    "max_output_tokens": 128,
    "store": true
  }'
```

The same endpoint works with OpenAI SDKs by replacing their base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-secret")
response = client.responses.create(
    model="qwen3.8-27b",
    instructions="Answer concisely.",
    input="What is speculative decoding?",
    max_output_tokens=128,
)
print(response.output_text)  # SDK helper derived from response.output
```

`output_text` is an SDK convenience property. It is not emitted as a top-level wire field; the
wire response contains typed `output` Items.

### Create request fields

| Field | NInfer Responses Core contract |
|---|---|
| `model` | required non-empty string; must equal the artifact-derived public model ID or explicit `--model-id` override |
| `input` | string or typed Item array; it may be omitted or empty only when `previous_response_id` already supplies a user query |
| `instructions` | optional string, inserted before the reconstructed conversation for this request only |
| `previous_response_id` | optional ID of a retained local Response |
| `max_output_tokens` | non-negative integer; omission executes with the model's `defaultMaxTokens` serve-config field (default `8192`) but remains `null` in the Response object |
| `stream` | boolean; `true` selects Responses SSE rather than a JSON body |
| `store` | boolean, default `true`; controls local retrieval and continuation state |
| `temperature` | finite number in `[0,2]` |
| `top_p` | finite number in `[0,1]` |
| `metadata` | at most 16 string pairs; keys at most 64 characters and values at most 512 |
| `client_metadata` | Codex client extension; an object or `null`, accepted as opaque tracing metadata with no generation effect |
| `reasoning.effort` | `none` requests disabled thinking; other standard effort values pass to the selected template |
| `chat_template_kwargs` | template parameters as a JSON object; standard options merge with typed fields |
| `preserve_thinking` | alias for `chat_template_kwargs.preserve_thinking`; conflicting values are rejected |
| `text.format` | `text` (default), `json_object`, or `json_schema`; see output constraints above |
| `tools` | direct function definitions or namespace groups containing function definitions; see below |
| `tool_choice` | `auto`, `none`, `required`, a named function, or function-only `allowed_tools` with mode `auto`/`required`; namespaced selection carries both `namespace` and `name` |
| `parallel_tool_calls` | `true` by default; `false` enforces at most one call |
| `max_tool_calls` | non-negative integer accepted as a hosted-tool no-op; NInfer does not execute hosted tools |
| `truncation` | omitted or `disabled`; overlong input fails instead of silently dropping Items |
| `top_logprobs` | omitted or `0` |
| `service_tier` | omitted, `auto`, or `default`; the response reports `default` |
| `background` | omitted or `false` |
| `include` | omitted or an empty array |
| `stream_options.include_obfuscation` | optional boolean; accepted as a transport hint, but this local server emits no padding |
| cache and client hints | `prompt_cache_key`, `prompt_cache_options`, `prompt_cache_retention`, and explicit breakpoints follow [OpenAI prompt caching](#openai-prompt-caching); `safety_identifier` and `user` are accepted as client hints |

Unknown top-level fields fail with `unknown_parameter`. Recognized but unsupported features fail
with a field-specific 400 error instead of being silently ignored.

### Input Item contract

String `input` is normalized to one user `message` with an `input_text` part. Array input accepts:

| Item | Supported form |
|---|---|
| `message` | roles `user`, `assistant`, `system`, and `developer`; string content or typed content array |
| `input_text` | message content part containing string `text` |
| `output_text` | assistant-message replay part containing string `text` |
| `refusal` | assistant-message replay part; its text enters assistant history |
| `input_image` | user- or assistant-message part with HTTP(S) or data-URI `image_url`; detail omitted or `auto`; requires the model's `vision` serve-config field |
| `input_video` | NInfer extension with HTTP(S) or data-URI `video_url`; requires the model's `vision` serve-config field |
| `reasoning` | raw replay Item with `reasoning_text` content; summary/encrypted metadata may accompany raw text but cannot replace it |
| `function_call` | completed assistant call with optional `id` and namespace, plus required `call_id`, `name`, and JSON-object string `arguments` |
| `function_call_output` | completed result with required `call_id` and optional matching name/namespace assertion; `output` may be a string or a non-empty array of `input_text`/`input_image` parts |

Contiguous assistant-owned Items form one assistant history turn in the representable order
`reasoning` -> assistant message content -> `function_call`. Multiple message Items append their
content parts, multiple calls retain declaration order, and a reasoning-only turn is retained. A
user, system, developer, or `function_call_output` Item ends the group; an order that would require
rearranging assistant content fails with `invalid_assistant_history`. Results are validated by
`call_id` and reordered to call declaration order before prompt rendering; unknown, duplicate, or
unrepresentable partial result sets fail with `invalid_tool_history`. Canonical input Items retain
client order. Input Item IDs are preserved when supplied and generated otherwise; duplicate IDs
fail.

System and developer message Items retain their positions in the input array. Top-level
`instructions` is represented as a leading developer turn for the current request; target-specific
role lowering occurs only in the Qwen family frontend.

An `input_text`, `input_image`, or tool-result part may carry
`prompt_cache_breakpoint:{"mode":"explicit"}`. Write selection follows
[OpenAI prompt caching](#openai-prompt-caching); boundaries affect reuse opportunities, not prompt
identity or output semantics. String message status/phase metadata is accepted but has no Qwen
prompt representation.

`input_file`, `input_audio`, image `file_id`, non-`auto` image detail, reasoning metadata without raw
reasoning text, partial tool Items, and other Item/content types are not supported. HTTP media URLs
stored in a response chain are fetched again when that chain is continued; use data URIs when the
historical media bytes must be immutable.

### Function tools

Responses function definitions may be declared directly rather than inside Chat Completions'
nested `function` object:

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Get current weather",
  "parameters": {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": false
  },
  "strict": true
}
```

They may also be grouped in a Responses namespace:

```json
{
  "type": "namespace",
  "name": "mcp__weather",
  "description": "Weather service",
  "tools": [{"type": "function", "name": "get_current"}]
}
```

NInfer gives each namespace/function pair a distinct internal Engine identity and restores the
separate `namespace` and `name` fields in aggregate output, SSE events, and replayed Items. The same
function name may therefore appear in different namespaces. Namespace members remain ordinary
client-executed functions; this does not add a remote MCP executor.

NInfer renders these definitions in the Qwen prompt and parses model output into separate
`function_call` output Items. Each output has a protocol Item `id` (`fc_...`) and a distinct
`call_id` (`call_...`). The client executes the function and sends a `function_call_output` Item in
a later request. Selection and strict argument enforcement follow the common tool contract above.

Hosted tools, remote MCP tools, custom free-form tools, deferred loading, output schemas, and
caller restrictions that exclude direct invocation remain unsupported.

### Response object and usage

A terminal wire response has `object: "response"`, one of `completed`, `incomplete`, or
`cancelled` in `status`, and a typed `output` array. NInfer may emit:

- a `reasoning` Item containing raw `reasoning_text` and an empty summary;
- an assistant `message` containing an `output_text` part;
- one or more `function_call` Items.

Ordinary model/string stops produce `completed`. Output-token or context-capacity exhaustion
produces `incomplete` with `incomplete_details.reason: "max_output_tokens"`. Errors accepted after
an SSE response has started produce `response.failed`; validation and preparation errors remain
normal HTTP error responses. `completed_at` is populated only for completed Responses. A
reasoning-only incomplete result contains no invented empty assistant message.

Usage is checkpoint-native:

```json
{
  "input_tokens": 42,
  "input_tokens_details": {"cached_tokens": 17},
  "output_tokens": 12,
  "output_tokens_details": {"reasoning_tokens": 5},
  "total_tokens": 54
}
```

`input_tokens` includes the chat template and expanded media tokens. `cached_tokens` is the exact
checkpoint-proven prompt prefix reused by Engine. `output_tokens` is the count of accepted generated token
IDs, including a withheld stop token when applicable. `reasoning_tokens` is counted in the Qwen
output decoder while accepted tokens are still in the reasoning channel; it is not estimated by
re-tokenizing decoded text.

### Responses streaming

Set `stream:true` for semantic Server-Sent Events. Every frame uses both the SSE event name and a
matching JSON `type`, and every JSON event has a monotonically increasing `sequence_number`:

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":7,...}

```

The normal lifecycle is:

1. `response.created`, then `response.in_progress`;
2. `response.output_item.added` and `response.content_part.added`;
3. zero or more `response.reasoning_text.delta` or `response.output_text.delta` events;
4. matching `*.done`, `response.content_part.done`, and `response.output_item.done` events;
5. exactly one `response.completed`, `response.incomplete`, or `response.failed` terminal event.

Function arguments use `response.function_call_arguments.delta` and `.done`. IDs, output indices,
and content indices remain stable, and concatenated deltas equal the terminal Item. Responses SSE
does not emit the Chat Completions `[DONE]` sentinel. With tools enabled, ordinary answer text still
streams immediately; only an ambiguous `<tool_call>` suffix or the structured tool region is held.
On the unconstrained route, malformed tool markup is flushed back as ordinary text without losing bytes.

### Local response state and resources

`store` defaults to `true`. Stored Responses live only in this server process and are bounded by an
LRU store. They are lost on restart and are not OpenAI's durable cloud retention service.

`previous_response_id` reconstructs the complete stored input/output Item history before the new
input. The current `instructions` value is placed first but is not saved into the continuation
context, matching the Responses rule that previous top-level instructions do not carry forward.
Function definitions are request configuration rather than conversation Items and must be sent
again on tool-result turns. The reconstructed prompt follows the ordinary Engine path, so compatible
checkpoint reuse applies naturally.

A stored Response also retains its resolved `preserve_thinking` value. A child which omits the
field inherits the parent value. An explicit different value creates a new semantic branch; prompt
rendering and identity still determine reuse. Changing the boolean alone never invalidates an exact
checkpoint already proved compatible by the model runtime.

For Engine-local reuse, a stored root Response receives one bounded session key derived from its
response ID, and every `previous_response_id` child inherits that key. `store:false` roots remain
anonymous; a `store:false` child may read its inherited session checkpoint but does not replace the
stored chain's latest endpoint. Response-store eviction or deletion removes the HTTP object, not an
independently retained Engine checkpoint; the latter remains bounded by the Engine's own retention
and pressure policy. No session key or cache marker is added to the HTTP schema.

Resource behavior:

| Endpoint | Contract |
|---|---|
| `GET /v1/responses/{id}` | returns the stored terminal object, or 404 `response_not_found`; stream recovery and non-empty `include` are rejected rather than ignored |
| `DELETE /v1/responses/{id}` | removes public retrieval and returns `response.deleted`; descendant contexts already retained by other Responses remain usable |
| `GET /v1/responses/{id}/input_items` | returns normalized Items supplied to that request; supports `after`, `limit` `1..100` (default `20`), and `order` `asc|desc` (default `desc`); image URLs are redacted unless `include=message.input_image.image_url` |
| `POST /v1/responses/{id}/cancel` | explicitly fails because background execution is unsupported |
| `POST /v1/responses/compact` | explicitly fails with `compaction_not_supported` |

`store:false` Responses cannot be retrieved or used as `previous_response_id`. LRU eviction and
explicit deletion also make an ID unavailable. A single Response larger than the configured store
capacity fails with `response_store_capacity_exceeded` rather than silently pretending it was
stored.

### Responses input token count

`POST /v1/responses/input_tokens` uses the same prompt path as Create and does not run generation.
It accepts `model`, `input`, `instructions`, `previous_response_id`, reasoning, function tools and
tool choice, supported text/truncation values, and the `preserve_thinking` extension. Parent lookup,
call-ID normalization, template rendering, and media expansion are therefore identical to the
corresponding Create request:

```bash
curl http://127.0.0.1:8080/v1/responses/input_tokens \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","input":"Count this prompt."}'
```

```json
{"object":"response.input_tokens","input_tokens":11}
```

Unsupported Create fields include Conversations, prompt templates, context management, hosted
moderation, non-empty `include`, background execution, compaction,
files/audio, and OpenAI-hosted/MCP/custom tools. These are compatibility boundaries, not silently
accepted placeholders.

## Anthropic Messages

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Explain prefix reuse in one sentence."}
    ]
  }'
```

The endpoint accepts top-level System text, ordered User/Assistant/System history, text and image
blocks, Thinking history, tool-use history, tool results, user-defined tools, aggregate responses,
and Anthropic SSE. Consecutive User or Assistant messages are joined without adding separators.
Mid-conversation System messages retain their input position. A final text-only Assistant message
is an Assistant prefill: generation continues its existing text instead of opening another turn.
Assistant prefill cannot contain media, Thinking, or tool calls and cannot start with Thinking
enabled.

Claude Code may place its attribution metadata in the first block of a top-level System array. If
that block is a text block beginning exactly with `x-anthropic-billing-header:`, NInfer consumes the
whole block before token counting, prompt preparation, and cache identity construction. The rule is
positional: a string-form System value, a later array block, or an inline System message with the
same text remains ordinary prompt content. A `cache_control` marker attached to the consumed block
is consumed with it rather than moved to adjacent content.

`max_tokens` is optional for local clients and otherwise uses the model's `defaultMaxTokens`
serve-config field (default `8192`); a positive
value is the complete output budget. `max_tokens:0` is rejected because NInfer does not expose a
completed zero-output cache-prewarm lifecycle. `temperature`, `top_p`, `top_k`, and
`stop_sequences` enter Engine execution. A matched custom stop is returned as
`stop_reason:"stop_sequence"` together with the actual `stop_sequence`; context exhaustion returns
`model_context_window_exceeded`.

Thinking supports `disabled`, `adaptive`, and `enabled`. Enabled Thinking requires
`budget_tokens >= 1024` and less than `max_tokens`, and that budget is passed to Engine. Visible
Thinking is returned with an opaque compatibility signature; SSE emits its `signature_delta`
before closing the block. Request lowering reconstructs the local prompt from the visible
`thinking` text and treats `signature` as non-semantic transport metadata, so retained history
remains usable across serve restarts.
`display:"omitted"` is rejected because NInfer cannot provide Anthropic's
encrypted hidden-reasoning restore semantics. `preserve_thinking` remains a NInfer extension for
closed-turn reasoning history. `output_config.effort` passes its protocol-validated value to the
selected template. `output_config.format` accepts JSON Schema output as described above.

User-defined tools support `name`, `description`, object `input_schema`, `input_examples`, and
`strict`. `tool_choice` accepts `auto`, `none`, `any`, or named `tool`; `disable_parallel_tool_use`
enforces a single-call limit. See the common tool contract above for schema and framing details.
Deferred tools, tools that exclude direct model calls, Anthropic-provided/server tools, toolsets,
MCP, and containers remain unsupported. `tool_result` preserves text/image order and marks
`is_error:true` explicitly in the model prompt. For a visible Assistant tool-use turn, the next
User turn must provide exactly one leading result for every declared ID; valid results are matched
by ID and normalized to call order. A history that begins with results remains valid as a truncated
or imported conversation.

Block-level ephemeral `cache_control` on tools and supported System/User/Assistant/tool-history
blocks creates explicit shared-prefix candidates. At most four distinct block-level breakpoints are
accepted. Request-level `cache_control` targets the last cacheable block: it merges with an explicit
breakpoint at the same target and TTL, conflicts at the same target with a different TTL, and needs
an available fifth slot when four different explicit targets already exist. TTL must be `5m` or
`1h`; it is a protocol hint, not a wall-clock residency guarantee.

NInfer maps representable boundaries to exact prompt frontiers and ignores a legal but
unrepresentable advisory boundary without changing the prompt. Reuse still requires exact rendered
identity and can read an existing owner without another `cache_control`. Aggregate usage reports
verified reused tokens in `cache_read_input_tokens` and leaves cache creation unknown. Streaming
emits `message_start` after Engine admission commits the prefix selection and before
transfer/prefill output, so its uncached/cache-read split is already exact; terminal cumulative
usage matches the aggregate response.

Documents, Search Results, Files, Structured Outputs, server-tool results, container uploads, and
other execution-dependent blocks are rejected with the missing capability identified. Metadata,
service tier, inference geography, protocol-version/beta headers, cache TTL, and unknown advisory
fields do not block an otherwise executable request. The request `model` is any non-empty local
proxy label and is echoed in the response; it does not select the resident artifact.

Every Messages response carries a `request-id` header; error bodies also carry `request_id` and use
Anthropic error categories. Local admission overload maps to HTTP 529 and queue/media timeouts to
HTTP 504. Streaming owns the full Anthropic block lifecycle for Thinking, text, and tool use.

`POST /v1/messages/count_tokens` uses the artifact's tokenizer, chat template, and media expansion
without generation. It shares the same prompt normalization, tools, Thinking mode, Assistant
prefill, media processing, and cache-marker interpretation as Messages; output-only sampling and
streaming fields do not affect the count:

```bash
curl http://127.0.0.1:8080/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Count this prompt."}]
  }'
```

## Decision scoring (POST /v1/decisions)

```bash
curl http://127.0.0.1:8080/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "The player holds a pair of aces and the pot is contested.",
    "questions": {
      "continue": {
        "type": "noul",
        "instructions": "Should the player continue?"
      },
      "action": {
        "type": "choice",
        "instructions": "Which action?",
        "criteria": {"raise": "Raise the bet", "fold": "Fold the hand"}
      },
      "strength": {
        "type": "score",
        "instructions": "Rate the hand strength.",
        "criteria": ["weak", "medium", "strong"]
      }
    }
  }'
```

Request shape: the context is exactly one of `state` or `messages` (a JSON-null `state` or
`messages` is not a supplied context; supplying neither is a 422). `state` is text-only — a
string is used verbatim and a JSON object/array is JSON-dumped UTF-8. `messages` is a chat
history: a non-empty array of `{role, content}` turns, where `role` is one of `system`,
`developer`, `user`, or `assistant` and `content` is a string (a single text turn) or an array
of parts, each part an object with a `type` of `text` (`{type, text}`) or `image_url`
(`{type, image_url}`); `image_url` is an HTTP(S) URL or a base64 data URI, given as a string
or an object `{url, detail?}` where `detail` is omitted or `"auto"`. `questions` is a map from
question id to `{type, instructions, criteria}`, where `type` is `noul`, `choice`, or `score`,
`instructions` is optional (default null, any JSON value), and the `criteria` rules follow the
type. `options` is an optional diagnostics object: the only supported key is `raw_logits`
(a boolean, default false); an unknown option field is a 422 (the field is named). When
`raw_logits` is true, each answer carries a `raw_logits` map (option to pre-softmax readout
logit) parallel to `probabilities`. The
request's optional `model` field (a string) is accepted and ignored: the route runs
on the model the router has loaded (the resident); with nothing loaded it answers 503
`model_not_ready`. The response's `model` field echoes the loaded model id.

An `image_url` part requires the model's `"vision": true` serve-config field; a media request
against a model without vision is a 400 with code `vision_disabled`.

- `noul`: optional map, keys restricted to `true`/`false` (any-JSON values);
- `choice`: map of 1-255 entries from key to description (any-JSON values), at most the
  artifact's compiled label-table size (<=255) entries, key order preserved;
- `score`: ordered array of 2-50 level descriptions (any-JSON values).

Unknown fields are a 422 (the offending field is named), including any field in a question
object other than `type`, `instructions`, and `criteria`, any field in a
message object other than `role` and `content`, and any `options` key other than `raw_logits`.

The response shape is `{model, answers{<id>: {type, ...}}, usage{input_tokens, output_tokens: 0}}`
with per-type answer fields:

- `noul`: the `noul` probability (P(true)) and no confidence field;
- `choice`: the winning `choice`, per-option `probabilities`, and `confidence` (the top option
  probability);
- `score`: `score` as the expected 0-based level index, a `legend` mapping level indices to their
  descriptions, per-level `probabilities`, and `confidence` (spread: 1 - sqrt(variance)/half_range
  over the level indices).

When the request set `options.raw_logits`, each answer also carries a `raw_logits` map (option
to pre-softmax readout logit) parallel to `probabilities`.

`usage.output_tokens` is always `0`: decision scoring generates no tokens. Errors: 401 for a
missing/invalid API key, 422 for body validation (the offending field is named), and 503 for a
model that is not loaded (or whose load failed), or for a decision job the loaded engine cannot
start (its shared working set cannot hold row 0 plus one branch row). Limits: choice questions
take 1-255 options (capped at the artifact's compiled label-table size, <=255; out of range is a
422) and score levels are 2-50.

The route runs on the model's loaded engine (no separate model load): the decision branch
workspace is allocated per job, sized to one round of branches, and the plan reserves a decision
region (two Device StateImage slots for row 0 plus one branch row, `D + 1` KV addresses, and
`D + 1` Main KV execution-table rows above the Generation lanes' rows), so chat traffic can
neither starve a decision job of capacity nor collide with its execution rows. A decision job is
serialized against in-flight generation rounds by the engine's execution lock. A question set
larger than one round is processed as successive fork / prefill / readout / release rounds, not
rejected: a round holds `min(free Device StateImage slots, free KV addresses, free decision
execution rows)` branches. Admission requires room for the shared row 0 plus one branch row, and
names the exact free counts when it fails, because with the reserved region in place a shortfall
means Generation traffic overran its configured axes. Every parsed decision request also appears in the operational
log and, when request logging is enabled, in the request JSONL: one `decision_start` before the
job runs and exactly one `decision_done` or `decision_error` terminal after it, carrying the
question and candidate counts, the answers, and the prefilled input tokens. A decision job's
prefilled tokens count toward the periodic throughput record. The full feature reference
(endpoint, contract, execution path, and binding design choices) is in
[Decision scoring](decisions.md); the model's execution path is in
[Decision scoring in the model reference](maintainer/qwen3_5-model.md#decision-scoring).

v1 semantics: each option is scored in isolation (a slice softmax over each question's candidate
set). Option-set interaction is limited to the shared slice denominator, so changing the option set
rescales the surviving options' probabilities; probabilities and confidence are uncalibrated slice
statistics (choice confidence = top option probability; score confidence = spread).


## llama.cpp / llama-ui compatibility

The bundled webui (`ggml-org/llama-ui`, a llama.cpp/llama-swap client) is served at `/` with
`--webui` and is most useful in ROUTER mode (multi-model via `--config`). It drives the
server-management routes below, which NInfer implements as **real, working functionality** backed
by serving-owned state: a server-side stream registry, a tool registry, live slot introspection,
a router-hooked model-status feed, and an owner-addressable cancellation token. These routes are a
**compatibility surface for that webui**, not part of the OpenAI or Anthropic protocol contract
documented above. NInfer ships no built-in server-side tools, and the live-generation behaviors
(a busy slot, a mid-stream cancel, a real load/unload transition) need a resident model on a real
GPU; without one, each route still answers with its real, structured response (an empty list, an
idle slot, `success: false`, or a well-formed error) rather than fabricating state. They are
key-gated by `--api-key` like the other API paths.

| Method and path | Behavior |
|---|---|
| `POST /models/load` | load/swap the resident model (ModelRouter) |
| `POST /models/unload` | force the no-resident state (ModelRouter) |
| `GET /models/sse` | router-hooked model-status SSE feed (live `model_status` events + keepalive) |
| `GET /slots` | live concurrency slot state (`is_processing`, `?model=`) |
| `GET /tools` | list registered server-side tools |
| `POST /tools` | execute a registered server-side tool |
| `POST /v1/streams/lookup` | list active (non-done) conversation streams |
| `GET /v1/stream/{id}` | replay + live-tail a registered stream (`?from=` byte offset) |
| `DELETE /v1/stream/{id}` | cancel + remove a registered stream |
| `POST /v1/chat/completions/control` | cancel an in-flight generation (`{"success":…}`) |

### Model load and unload

- `POST /models/load` — body `{"model": <id>}`. Runs the in-process router's on-demand load/swap.
  A swap first DRAINS the evicted resident's live in-flight Grants AND its live VRAM handle
  pins: both keep the backend (its weights, KV cache, and CUDA graphs) and its VRAM alive, and
  the drain waits on both reaching 0, bounded by the `healthCheckTimeout`. A Grant without a
  live handle (granted but not yet submitted, or whose handle was already released) keeps the
  evicted backend + Engine alive, so draining the pin count alone would let the swap construct a
  second Engine while the old model's VRAM is still resident → OOM. Only then is the resident
  destroyed (the VRAM is actually freed) and the target loaded. This is the single-GPU,
  one-resident, sequential-swap model — no concurrent residency and no preemption of in-flight
  requests: on a drain timeout the swap is deferred (the resident is restored intact) and the
  request is rejected.
  `extra_args` and other body fields are ignored (model config is fixed at registration).
  `200 {"model":<id>,"status":"loaded"}`; `400 model_required` when `model` is missing;
  `404 model_not_found` for an unknown id; `503 model_not_ready` when the load/swap does not
  reach a ready state (a failed load, a readiness timeout, a drain timeout with the evicted
  resident's in-flight grant or VRAM pin still live, or a full swap-join queue). This
  endpoint only triggers the load/swap and never does request-concurrency admission: the
  429 `server_overloaded` / 503 `request_queue_timeout` for in-flight generation requests
  originate from the engine layer (bounded FIFO ingress, `--max-pending-requests`), not from
  the router.
- `POST /models/unload` — body `{"model": <id>}`. Forces the no-resident state: the unload
  DRAINS the resident (bounded by the `healthCheckTimeout`) before forcing the no-resident state,
  and a named model that is not the loaded one still returns `200`. On a drain timeout the
  resident is restored intact and the request is rejected with `503 model_not_ready`.
  `200 {"model":<id>,"status":"unloaded"}`; `404 model_not_found` for an unknown or missing
  `model`.

### Model-status feed

- `GET /models/sse` — `Content-Type: text/event-stream`. The feed is **router-hooked pub/sub**:
  on every real `ModelRouter` transition (load, swap-ready, explicit unload, or TTL eviction) the
  router emits a status event, and the hub fans it out to all connected clients as a live
  `model_status` record. Idle subscribers also receive periodic `: keep-alive` comments on a 5 s
  cadence. Delivery is non-blocking, and a client that cannot drain its queue is dropped so one
  slow reader never stalls the router's event path. Record shape:

  ```json
  {"event":"model_status","model":"<id>","data":{"status":"<status>","exit_code":0}}
  ```

  `data.status` is `unloaded`, `loading`, or `loaded` (the router's live state). While
  `status` is `loading`, `data.progress` reports the Engine's real model-load byte progress
  as a 0..1 fraction (bytes loaded so far / total bytes, from the Engine's startup observer),
  so it advances during a load; on `loaded` and `unloaded` records it is `0.0`. With no resident model the feed opens with an
  `unloaded` record and then holds the connection with keepalives.

### Live slot state

- `GET /slots` — `200` with a JSON array of **`max_concurrency` slot objects**, one per
  configured concurrency slot; `?model=<name>` filters to that model's slots. Each slot:

  ```json
  {"id":0,"model":"<id>","is_processing":true,"state":"decode"}
  ```

  `is_processing` is `true` for a slot holding an in-flight generation and `false` otherwise,
  computed from the router's live in-flight count, the resident model, and the backend's
  `runtime_stats()`. `state` is an informational `idle` / `prefill` / `decode` / `loading`
  label. With no resident model the array reports `max_concurrency` idle slots
  (`is_processing:false`, `state:"idle"`, empty `model`) — the correct live state, not an empty
  list.

### Server-side tool registry

- `GET /tools` — `200` with a JSON array of the registered tools, each `{"name":<name>,
  "definition":<json schema>,"enabled":true}`. NInfer ships **no built-in server-side tools**
  (it is a client-side function-calling engine), so the registry starts empty and this returns
  `200 []`.
- `POST /tools` — body `{"tool":<name>,"params":<object>}`. **Executes** the named tool through
  the server-side dispatcher and returns `200` with `{"plain_text_response":<result>}` on
  success or a structured `{"error":"<message>"}` on failure. For a tool that is not registered
  the dispatcher answers `{"error":"tool '<name>' is not available on this server"}`. The
  dispatch machinery is real; only the default tool set is empty.

### Conversation-stream registry

NInfer keeps a **server-side `StreamRegistry`** — its first durable server-side state — keyed by
the stream id the webui sends in the `X-Conversation-Id` header (`conversationId::model`). Each
in-flight *streaming* chat completion registers an entry whose raw SSE bytes are buffered, so a
stream can be re-attached by byte offset and tailed live from another request.

- `POST /v1/streams/lookup` — body `{"conversation_ids":[<id>, …]}`. `200` with a JSON array of
  the **active (non-done)** matching streams, each `{"conversation_id":<id>,"is_done":false,
  "started_at":<epoch s>}`; `200 []` when none of the ids are active.
- `GET /v1/stream/{id}` — `Content-Type: text/event-stream`. Replays the buffered SSE from the
  byte offset in `?from=` (default `0`), then **live-tails** new frames until the stream is done
  and closes. An unknown `{id}` returns `404` with `{"error":{"code":"stream_not_found"}}`.
  This endpoint deliberately emits no keepalive comments (unlike the generation SSE endpoints):
  `?from=` is a byte offset measured against the generation stream's frames, and injecting
  keepalive bytes would desync the resume offset. While live-tailing, an idle generation sends
  no bytes at all — a quiet connection means "waiting for the next frame", not a dead socket,
  so the client must handle idle itself; it cannot rely on a periodic heartbeat to keep the
  connection alive.
- `DELETE /v1/stream/{id}` — removes the stream and, if its generation is in flight, cancels it.
  `200` with `{"cancelled":true}` when the stream was found and removed, `{"cancelled":false}`
  when the stream was unknown (never registered, or already reaped past its retention window); a
  done-but-not-yet-reaped stream is still found and removed, so it reports `{"cancelled":true}`.

### Reasoning control (cancellation)

- `POST /v1/chat/completions/control` — body `{"id":<completion id or stream id>,"action":
  "reasoning_end","model"?}`. Signals the matching in-flight generation to **stop now**, via the
  per-stream cancel token that composes into the `CancellationView` channel the Engine already
  polls between decode rounds. Returns `200` with `{"success":true}` when the id resolved and the
  generation was signalled, or `{"success":false}` when the id is unknown. This is an
  owner-initiated stop of one request's generation — **not** scheduler preemption: it touches no
  other request, reorders no FIFO, and evicts no resident model.

### Under the hood

Two serving-owned additions back these routes: a `StreamRegistry` (durable, keyed state that
outlives a single HTTP response) and a per-generation cancel token held by each registry entry and
OR-composed into the existing `CancellationView`, whose decode-loop observation point is
unchanged. Both live entirely in the serving layer (`src/serve/`); the Engine itself is not
modified.

## Authentication and CORS

Pass `--api-key VALUE` to require the same value as an OpenAI bearer token or Anthropic
`x-api-key` header. `GET /health` and CORS preflight requests remain unauthenticated.

```bash
curl http://127.0.0.1:8080/v1/models \
  -H 'Authorization: Bearer local-secret'
```

`--cors` adds permissive browser CORS headers. It is disabled by default.

## Server options

The table lists executable CLI defaults. Per-model engine presets (context, KV, speculative,
prefill, vision, output limit) live in the serve-config JSON (the field table follows the CLI
options); the startup example selects a long-context FP8/MTP3 profile.

| Option | Meaning | Default |
|---|---|---:|
| `--config FILE` | multi-model serve config: the model list plus per-model engine presets (required) | none |
| `--host H` | listen address | `127.0.0.1` |
| `--port N` | listen port | `8080` |
| `--api-key KEY` | required bearer or `x-api-key` value | unset |
| `--model-id ID` | override the public OpenAI model alias | artifact `metadata.name`, or architecture name |
| `--max-context N` | logical context ceiling of each sequence | `8192` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `8192` |
| `--max-concurrency N` | the single concurrency setting: maximum in-flight requests per model (the router's admission gate and the engine decode-batch count); valid range `1..8` | `1` |
| `--max-pending-requests N` | additional requests allowed to wait for admission | `16` |
| `--pending-timeout-ms N` | maximum preparation-plus-admission wait | `30000` |
| `--log-stats-interval-ms N` | aggregate throughput report interval; `0` disables it | `5000` |
| `--log-level trace\|debug\|info\|warning\|error\|critical\|off` | pretty stderr verbosity | `info` |
| `--device N` | CUDA device index | `0` |
| `--context-cost-presets FILE` | optional runtime context-cost preset registry | generic + compiled defaults |
| `--max-request-mib N` | body-size limit before JSON parsing | `384` |
| `--media-cache-mib N` | LRU-retained prepared BF16 media payloads; `0` disables retention | `1024` |
| `--media-live-mib N` | all live prepared BF16 media payloads | `2048` |
| `--media-preprocess-threads N` | bounded media preprocessing workers; `0` selects at most 16 from host concurrency | `0` |
| `--request-log-jsonl FILE` | append full-precision server/request records | disabled |
| `--response-store-max-records N` | maximum locally retained Responses objects | `1024` |
| `--response-store-max-mib N` | total local Response envelope/Item/context budget | `256` |
| `--default-thinking-budget N` | positive thinking cap inherited by thinking-enabled requests | unset |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--no-prefix-reuse` | disable compatible-prefix caching | prefix reuse on |
| `--device-state-slots N` | extra Device StateImages beyond `max-concurrency` | `max-concurrency` |
| `--host-context-mib N` | shared pinned Host budget for StateImages, KV and pause snapshots, including in-flight destinations | `8192 MiB + 8 native StateImages` |
| `--no-thinking` | disable thinking by default | thinking on |
| `--preserve-thinking` | preserve closed-turn assistant reasoning by default | off |
| `--cors` | permissive browser CORS headers | off |
| `--temperature F` | process-level temperature override | unset |
| `--top-p F` | process-level top-p override | unset |
| `--top-k N` | process-level top-k override (`0..20`; zero selects the top-20 cap) | unset |
| `--min-p F` | process-level min-p override | unset |
| `--presence-penalty F` | process-level presence-penalty override | unset |
| `--frequency-penalty F` | process-level frequency-penalty override | unset |
| `--seed N` | fixed seed when a request omits one | fresh random seed per request |
| `--greedy` | force exact argmax for all requests | off |

Per-model engine presets are set per model entry in the serve-config JSON (`--config FILE`), not
on the command line. A model entry without a field uses the shared default; a field present on
one model never affects another. Top-level config fields:

| Field | Meaning | Default |
|---|---|---:|
| `models` | object mapping each public model ID to its entry; at least one entry required | none |
| `healthCheckTimeout` | seconds the router waits for the evicted resident's in-flight grant + VRAM-pin drain and the loaded model's readiness gate | `120` |
| `globalTTL` | idle seconds before a loaded model is unloaded; a model's own `ttl` wins when `> 0`; `0` disables auto-unload | `0` |

Per-model entry fields (all optional; an absent field uses the default):

| Field | Meaning | Default |
|---|---|---:|
| `artifact` | path to the `.ninfer` artifact | required |
| `identity` | the model ID the Engine reports; also the key's default | the entry key |
| `aliases` | extra public IDs that resolve to this model | `[]` |
| `ttl` | idle seconds before this model is unloaded; `0` falls back to `globalTTL` | `0` |
| `maxContext` | logical context ceiling of each sequence of this model | `8192` |
| `defaultMaxTokens` | output limit when a request omits `max_tokens`/`max_output_tokens` | `8192` |
| `kvCapacity` | explicit shared Main Text KV token count, or `"auto"` to maximize it from remaining GPU memory; omitted follows `maxContext` | `maxContext` |
| `kvDtype` | KV-cache storage: `bf16`/`int8`/`fp8`/`nvfp4`/`k8v4` | `bf16` |
| `spec` | speculative backend `mtp`/`dflash`/`dflash2`; omitted loads no speculative backend | off |
| `draftTokens` | MTP `1..5`; DFlash/DFlash2 `1..15` (used only with `spec`) | unset |
| `lmHeadDraft` | optimized proposal head (used only with `spec`) | off |
| `prefillChunk` | text-prefill chunk (positive multiple of 128) | `1024` |
| `vision` | enable media input and load the model's Vision GPU allocations | off |

The router logs the complete preset set of a model (identity, the normalized engine parameters,
and which fields the config set) at each on-demand load or swap-in.

Context-cost coefficients resolve once at startup from generic defaults, matching compiled values,
and optional transfer or prefill entries from `--context-cost-presets FILE`. Prefill entries match
the hardware and a signature derived from the actual Text/Vision configuration, bindings and Uses.
A new representation without a matching measurement uses generic prefill coefficients. A malformed
file aborts startup; the operational context-cost record and JSONL `server_start` identify the
selected source.

Engine selects sampling defaults from the loaded architecture and the request's resolved thinking mode.
Qwen3.6-27B and Qwen3.8-27B use `1.0/0.95/20/0/0` for
temperature/top-p/top-k/min-p/presence penalty in thinking mode and `0.7/0.80/20/0/1.5` in
non-thinking mode. Qwen3.6-35B-A3B differs only in its thinking presence penalty, which is `1.5`.
Frequency penalty is `0` for all registered presets. Process flags override registered values,
request fields override process flags, and `--greedy` finally forces temperature `0`.

For `C=--max-concurrency` and `H=--device-state-slots`, total Device StateImage capacity is `C+H`.
Host state and Main/Backend KV share one startup-fixed byte budget; this is context storage, not a
limit on total process RAM. `--host-context-mib 0` disables Host context backing.
`--no-prefix-reuse` disables cross-request history reads and writes; pause/replay recovery remains
available, and the capacity flags may still be specified.

Run `./build/apps/ninfer-serve --help` for the exact option contract.

Serve writes human-readable operational records to stderr using
`YYYY-MM-DD HH:MM:SS.mmm  LEVEL  message`. Normal output covers material startup milestones,
readiness, request lifecycle, fixed-interval throughput, and shutdown; `--log-level debug` exposes
internal startup and resource-planning detail. A terminal may use one transient line during startup,
but Serve throughput is always a persistent record. Redirected stderr contains no terminal control
sequences. Pretty values use readable units and rounded rates; use the independent request JSONL for
complete fields and full precision. Operational records never contain prompts, generated text,
request bodies, credentials, or arbitrary client error messages.
If a tool marker is returned to text because its structure or tool identity cannot be represented,
Serve emits one warning with only the failure classification, never the generated markup.

## Live metrics

`GET /metrics` serves Prometheus text format 0.0.4 on the same port. It follows the server's
API-key authentication and works independently of `--request-log-jsonl` and
`--log-stats-interval-ms`. Engine failure leaves the endpoint readable with
`ninfer_engine_ready 0`. Counters start after startup warmup and reset when the server restarts.

```bash
curl http://127.0.0.1:8080/metrics
```

| Metrics | Meaning |
|---|---|
| `ninfer_model_info`, `ninfer_max_concurrency`, `ninfer_max_context_tokens` | Model/backend identity and startup limits |
| `ninfer_requests_running`, `waiting`, `paused`, `prefilling`, `decode_ready`, `replaying`, `materializing` | Current Engine gauges; prefill/decode/replay are subsets of resident requests |
| `ninfer_prompt_tokens_total`, `ninfer_prompt_tokens_cached_total` | Full input and reused tokens counted once on initial binding |
| `ninfer_prefill_tokens_total`, `ninfer_replayed_tokens_total` | Actual initial prefill and separate recovery recomputation |
| `ninfer_generation_tokens_total`, `ninfer_decode_tokens_total` | All committed outputs, or decode/control outputs excluding the first token; include thinking and injected control tokens |
| `ninfer_spec_decode_{rounds,draft_tokens,accepted_tokens,fallback_steps}_total` | Live native speculative work, including MTP and DFlash/DFlash2 |
| `ninfer_{preemptions,snapshot_restores,replay_restores}_total` | Pressure pauses and recovery routes |
| `ninfer_device_kv_{used,capacity}_pages`, `ninfer_device_state_{used,capacity}_slots` | Physical Main KV and StateImage occupancy; retained history also occupies these pools |
| `ninfer_host_context_{used,reserved,capacity,peak}_bytes` | Unified Host backing; reserved bytes are already included in used bytes |
| `ninfer_context_transfer_bytes_total{resource,direction}` | Actual State/Main KV/backend KV payload transfers |
| `ninfer_host_work_seconds_total{phase}`, `ninfer_device_wait_seconds_total` | Instrumented worker wall time; device wait is not CUDA kernel time |
| `ninfer_constraint_requests_total{outcome}`, `ninfer_constraint_cache_total{result}` | Settled constrained requests by completion state and compilation-cache access |
| `ninfer_constraint_{prepare,mask,matcher}_seconds_total`, `ninfer_constraint_mask_{positions,upload_bytes}_total` | Constraint work aggregated at request settlement, including truncated/cancelled results |
| `ninfer_constraint_draft_wait_seconds_total` | Live draft-ready wait counted once per batch; a subset of device wait |
| `ninfer_requests_total{outcome}`, `ninfer_response_failures_total` | Generation attempts entering preparation and subsequent response failures; protocol/model validation failures and token-count requests are excluded |
| `ninfer_time_to_first_token_seconds` | Histogram updated once at the first committed token, including preparation, queueing and binding |
| `ninfer_request_duration_seconds`, `ninfer_request_queue_seconds` | Histograms for settled generation outcomes, including cancellation; exceptional failures have separate counts |

Histograms expose `_bucket`, `_sum` and `_count`. Metrics have bounded labels; they do not retain
request IDs or request text. Rates are calculated by the consumer, for example:

```promql
rate(ninfer_generation_tokens_total[1m])

rate(ninfer_spec_decode_accepted_tokens_total[1m])
/ rate(ninfer_spec_decode_draft_tokens_total[1m])
```

The handler copies published snapshots and formats them outside the Engine worker. Scraping does
not reset counters or initiate device work. Detailed per-request records remain available through
the JSONL log. Monitoring tools with engine-specific metric names need an NInfer adapter.

## Structured request log

`--request-log-jsonl FILE` enables the machine-readable measurement log. The server opens `FILE`
in append mode and flushes every event, so successive model or MTP blocks may share one campaign
file. The parent directory must already exist. Failure to open the file aborts startup; the log path
is also rejected if it resolves to the model artifact.

Every line is one `ninfer_serve_request_log` schema-v25 JSON object. All events carry
`timestamp_unix_ms` and a process-unique `server_instance_id`; request IDs are monotonic only within
that server instance. Successful request-start records include request-scoped acquisition,
media-preprocessing wall/work, tokenizer, cache hit/miss/single-flight, and payload-size fields;
they do not infer request behavior from process-global counter deltas.

| Event | Contents |
|---|---|
| `server_start` | artifact path, architecture, public name, actual formats and prefill signature; resolved Engine and context-cache capacities, thinking/non-thinking sampler defaults plus process overrides, thinking-history and thinking-budget defaults, Device arenas, the optional non-additive Vision layout inside the unified workspace, unified Host context capacity and occupancy, KV sizing ledger, CUDA Graph allowance, CUDA/GPU environment, and redacted argv |
| `request_start` | protocol, resolved sampler and seed, requested reasoning effort, actual initial thinking mode and optional budget, Responses semantic-change flag, output budget, stream/message/tool shape |
| `request_rejected` | parsed request shape, requested reasoning effort, media-item count, `phase: "prepare"`, and the exact HTTP status/type/code/parameter/message for a synchronous preparation rejection |
| `request_done` | finish reason, prompt/completion/cache/computed-prefill tokens, prefix reuse path, tool-call parse diagnostics, preemption/recovery counters, thinking-budget application counters, unrounded request-stage seconds, per-request Engine Host exposure, and complete speculative-decoding counters |
| `request_scheduling` | request identity, pause/restore/recovery transitions, Snapshot revocation, Engine observation time and cumulative global/request work counters |
| `request_error` | the resolved request configuration and the generation, cancellation, or pre-outcome transport terminal message |
| `throughput` | interval token/decode/context-cache pressure counter deltas, authoritative worker Host-work deltas, current scheduler/resource gauges, and decode-round batch statistics |

`requested_reasoning_effort` and `preserve_thinking` record the explicit options, or `null` when
unspecified. `enable_thinking` records whether the response starts in thinking mode.

`request_done.result.tool_call_parse` records whether a complete marker was seen, the structured
call count, empty non-string arguments omitted during normalization, schema-mismatched arguments
preserved for consumer validation, and a stable text-fallback reason. Fallback reasons are `none`,
`malformed_structure`, `duplicate_parameter`, `invalid_tool_name`, `undeclared_tool`, and
`trailing_content`. These counters contain no tool arguments or generated text.

`request_done.constraint` carries the same constraint observation as the HTTP terminal result,
or `null` for unconstrained requests. Preparation failures and execution errors use the existing
rejection/error records rather than successful constraint outcomes.

`request_done.timings_seconds` contains `prepare`, `ttft`, `vision`, `prefill`, `decode`, and `total`
as full-precision JSON numbers. Its `speculative` object contains `backend`, `draft_window`, `rounds`,
`drafted_tokens`, `accepted_tokens`, `fallback_steps`, and `accepted_per_position`. Rates can be
derived downstream from raw token counts and seconds instead of rounded stderr strings.
`generation.scheduling` records preemptions, snapshot/replay restores, replayed tokens, paused time
and request-owned transfer bytes. Replay rebuilds committed state without adding new output usage.

`generation.admission` records the initial `preferred_reused_tokens`, `source_wait_seconds`,
`revoked_checkpoints`, and `fallback_reason`. Source waiting is a subset of initial queue time;
selecting or retaining a checkpoint does not itself count as a cache hit. Revocations count retained
checkpoint references removed under resource pressure. Fallback reasons are `none`, `source_invalid`,
`source_revoked`, `cost_changed`, `capacity_limit`, and `isolated_capacity`.

`request_scheduling` records `pause_started`, `paused`, `restore_started`, `restored`,
`replay_complete`, `recovery_complete`, `snapshot_revoked`, and a `terminal` boundary for preempted
requests. `preemption_index` identifies each pause cycle; `route` is `snapshot`, `replay`, or `null`
while pause preparation has not yet selected the saved representation. `steady_ns` is captured on
the Engine worker, and `elapsed_ns` starts at Engine submission; JSONL writes happen on the request
consumer thread. Compare `steady_ns` rather than delivery order across requests. `restored` ends
binding; `recovery_complete` marks the first fresh committed unit or normal terminal progress,
not merely rebuilding the old frontier. Cancellation can end the cycle without that event.

Each event's `progress` carries global and request-owned prefill, decode/control and replay token
counters. Between two boundaries, subtract the request delta from the global delta to measure
other requests' completed work. In particular, `restored` to `replay_complete` establishes whether
other work advanced during Replay without relying on periodic scheduler gauges. Events are enabled
only with request logging and add no per-token records.

For `server_start.memory`, `workspace.capacity_bytes` is the only physical workspace allocation.
When Vision is enabled, `vision_workspace` reports the aggregate prompt and maximum-item token
bounds plus encode peak and handoff layout/usage within that same allocation; these bytes must not
be added to `workspace.capacity_bytes`. The field is `null` when Vision is disabled.

`request_done.engine_timing` separates FIFO `queue_wait_seconds`, blocking
`device_wait_exposed_seconds`, and five mutually exclusive Host-active exposure phases under
`host_exposed_seconds`: `engine_boundary`, `program_submit`, `program_post`,
`engine_commit_output`, and `engine_maintenance`. `total` is exactly their sum and excludes Device
wait. The nested `decode` object reports the request's decode-class Host exposure, Device wait, and
round count; `units` reports its prefill/control unit counts. In a compact batch every participating
request is delayed by the full round, so these values explain request latency but **must not be
summed across concurrent requests**.
`constraint_draft_wait_exposed_seconds` is the request's exposure to the batch's draft-ready wait,
already included in `device_wait_exposed_seconds`. The `throughput.host_work.constraint_draft_wait_seconds`
interval and Prometheus counter count each batch once.

`request_done.first_output_timing` freezes observations immediately before Engine publishes its
first nonempty output delta. It is `null` when no such output exists. This boundary differs from the
first accepted model token and the client's first HTTP output. Engine elapsed time begins at submit:
initial queue ends when the successful binding attempt starts, initial binding ends when the
binding is installed, and paused time includes pause preparation, waiting and restoration. The remaining
interval is resident time. Its `engine` observations describe resident Host/Device-wait exposure;
`prefill` and `replay` describe this request's submitted work. Terminal `engine_timing` still covers
the whole request.

The work `gpu_seconds` measures Text prefill stream intervals, including their MTP/DFlash work;
Vision encode, standalone bridges and exact-hit sampling fall outside that interval.
`context_transfers` reports completed request-owned State/Main KV/backend KV copies by direction,
using transfer-event time and bytes. GPU intervals overlap Host submission and waits, so they are
separate evidence, not additional wall-time stages. Background reclamation remains Engine-wide.

`result.generated_token_ids` records committed output token IDs for exact prefix analysis.
The JSONL file never records an API-key value; `argv` replaces it with `<redacted>`.
Operational stderr summaries are rounded and are not the
aggregation source. OpenAI Responses, OpenAI Chat, and Anthropic generation requests receive a
request ID when they enter synchronous preparation. Successful preparation produces
`request_start`; a preparation failure produces `request_rejected` without a matching start. Each
started generation transaction then has exactly one machine terminal: `request_done` when Engine
returns its outcome, or `request_error` when generation fails before an outcome exists. Later
response rendering, Responses storage, or terminal transport failures are operational response
events only and do not add a second JSONL terminal. Schema/model validation rejections before
preparation and token-count-only calls are not measurement requests and do not receive request IDs.

By default the server persistently reports aggregate activity every five seconds. `prefill` counts
prompt suffix tokens actually computed during the interval, excluding prefix-cache hits; `decode`
counts tokens finally committed by decode rounds, excluding the first token produced by prefill.
For MTP, DFlash and DFlash2 this is the accepted committed output, not draft or rejected tokens.
Pretty `batch` and JSONL `average_size` are decode row-rounds divided by decode rounds during the
same interval. The
`running`, `prefilling`, `decode_ready`, `waiting`, `paused`, `replaying`, `materializing`,
`capture_pending`, and `terminal_pending` fields are the Engine scheduler snapshot at the end of the
interval. The JSONL `context_cache` object reports selections, captures, StateImage operations,
transfers, tail-page COW and pressure spills as interval deltas; `occupancy` and `last_selection` are
end-of-interval gauges. The separate `scheduling` object reports preemptions, restores and replayed
tokens. Occupancy includes Host reservations while transfers are in flight.

The JSONL `throughput.host_work` object is the aggregation authority: the Engine worker counts each
wall-time segment once, independent of batch size. `elapsed_seconds` contains the same five
mutually exclusive Host phases and their `total`; `device_wait_seconds` is separate.
`work_class_seconds` splits Host and Device-wait time into decode, prefill, and control classes.
`detail_subset_seconds` and `detail_invocations` expose stats-publication work; these detail values
are already contained in a top-level Host phase and must not be added to `total`.
Per-round, per-row-round, and per-invocation normalized values are
`null` when their denominator is zero. Pretty throughput contains nonzero token rates and counts,
the current running/prefill/decode-ready composition, nonzero waiting/materialization/terminal
states, average decode batch, and Host-active time plus its fraction of the interval. Use JSONL for
complete measurement analysis.
Intervals with context materialization or retention activity are retained even when they contain no
token execution; only fully idle intervals are omitted. Downstream measurement should prefer the
raw counters and seconds over rounded stderr rates.

## Execution behavior

The server owns one resident Engine with `1..8` execution lanes fixed at startup. At each decode
boundary, eligible decode-ready requests form one compact batch, processed by one model traversal
and, when graphs are enabled, one exact-batch CUDA Graph replay. A
request joins that batch only after its single-request prefill finishes; when it completes or is
cancelled, the next boundary rebuilds the batch without an empty row.

`--max-pending-requests` bounds the requests waiting behind the active set. The total generation
request lifetime capacity is `max_concurrency + max_pending_requests`, including requests still in
CPU/media preparation and completed model results whose response has not yet been released. A full
capacity returns HTTP 429 with code `server_overloaded`. The absolute
`--pending-timeout-ms` deadline starts before preparation, covers media acquisition and Engine FIFO
waiting, and returns HTTP 503 with code `request_queue_timeout` if admission does not occur in time.
Once admitted, a request may pause for resource pressure without restarting this initial-admission
deadline. Fresh requests enter in FIFO order with a bounded bypass allowance when an earlier request
cannot fit. There is no admission ETA or unbounded overflow queue.

Input memory is bounded by the outstanding-request count and the per-request
`--max-request-mib` limit. Media preparation uses a shared permit pool sized from `--media-live-mib`
and the maximum supported prepared-payload size per request. Waiting media requests retain the same
cancellation and timeout deadline. Model output is bounded by the same finite request count and
each request's effective output-token limit; output callbacks and network serialization run
outside the GPU executor and do not delay formation of the next batch.

`maxContext` (the model's serve-config field) is each sequence's logical ceiling. `kvCapacity`
fixes the shared Main Text KV pool used by active requests and retained prefixes. `auto`
accounts for the complete enabled runtime and leaves 1 GiB of sizing headroom; omitting the field
makes it follow `maxContext`. Capacity resolves once at the model's load.

Before each prefill, decode or replay unit, the runtime reserves the additional pages and temporary
storage required by that unit. It first reclaims inactive cache resources when capacity is short.
If resident requests still cannot advance together, it pauses a younger request while preserving
progress for the oldest resident request. A paused request does not block fresh requests that fit
the remaining capacity. Restoration follows original request order and reserves enough space to
rebuild the saved frontier and complete one new execution unit.

A paused request keeps its committed output and protocol state. With sufficient Host backing, it
can restore a snapshot; otherwise it rebuilds model state from retained input and committed tokens.
Replay does not resample or republish those tokens, but it consumes compute and can increase gaps in
the output stream. The policy and ownership rules are defined in
[Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md).

Compatible prefixes are reused for both text and multimodal histories unless the server starts with
`--no-prefix-reuse`. A multimodal hit additionally requires matching token types, three-axis MRoPE
positions, encoded-media digest, grid, and consumer spans. Media wholly inside a matched prefix
skips Vision execution, while new suffix media is encoded normally. The pretty completion record
shows `cache N (P%, path)` using readable path labels; JSONL retains the exact
`prefix_cache_hit_tokens` and `prefix_reuse_path` fields (`root` or `checkpoint`). Reuse requires
matching KV, recurrent state, hidden state, selected-backend state and exact prefix identity.

Completed conversation endpoints serve direct continuations. A separate input checkpoint preserves
the stable boundary before a response that a later prompt may normalize or replace. With
`preserve_thinking=true`, that boundary precedes the response's generation prologue; with `false`,
it precedes the assistant turn whose closed reasoning may be omitted. The next request can therefore
recompute the changed suffix while retaining the preceding conversation. Capture follows these
semantic boundaries and shared-prefix hints.

Changing `preserve_thinking` or reasoning effort changes the rendered prompt where applicable;
already retained exact prefixes remain usable. An appended mid-conversation system message is an
ordinary suffix. Modifying, removing or moving a historical message changes the prefix and may
require an earlier checkpoint or a root prefill.

Speculative backends preserve protocol output shapes, stop behavior, and usage accounting. If a stop
truncates a multi-token MTP, DFlash or DFlash2 round, the Engine commits the exact accepted target prefix so
a following compatible turn can reuse it. Output-limit and context-capacity finishes map to
`length`/ `max_tokens`; ordinary model or string stops map to `stop`/ `end_turn`.

Function tools are rendered into the model prompt and generated calls are parsed into protocol
responses. NInfer does not execute tools or enforce tool-argument schemas through constrained decoding.

Prompt-token usage includes chat-template and expanded media tokens. Generated-token usage comes
from accepted output token IDs, including a stop token whose decoded text may be withheld.
