# NInfer (personal fork)

This repository is a personal fork of
[upstream NInfer](https://github.com/Neroued/ninfer) with end-user serving additions. The
repository is experimental: changes are subject to being wiped without notice based on the
maintainer's own usage observations. The sections below the fork section describe the shared
upstream product.

## What this fork adds

### Multi-model router

The server runs an in-process multi-model router. It adapts llama-swap's router to own an
in-process Engine lifecycle instead of a child-process one. The router starts no-resident:
no model loads at startup and no warm-up runs. The first request for a model loads it on
demand. A request for a different model swaps the resident: it destroys the loaded Engine
to free VRAM, builds the target Engine, and gates on readiness under a health-check
timeout. A 1-second TTL ticker unloads an idle model once it has been idle past its
effective TTL and holds no in-flight request; an in-flight request pins the model. A
serve-config JSON (`--config`, required) names the model list and each model's engine
presets. Load, swap, and unload transitions publish live `model_status` events on a
router-hooked SSE feed (`GET /models/sse`). The swap lifecycle also exposes the llama.cpp
compatibility routes `POST /models/load` and `POST /models/unload`. The router does not
cap request concurrency; the engine layer owns bounded FIFO ingress. See [HTTP
serving](docs/serving.md).

### Decision scoring (JEV-alike)

`POST /v1/decisions` (alias `POST /v1/systemone`) scores a set of typed questions over a
shared state without generating tokens, so `output_tokens` is always 0. The context is
either a `state` string or a `messages` chat history (with optional image parts). The
questions are a map from id to `{type, instructions, criteria}`. Three types run:

- `noul` — a true/false question; the answer is the P(true) probability.
- `choice` — a named option set of 1-255 entries; the answer is the winning key,
  per-option probabilities, and a normalized Gini confidence.
- `score` — an ordered level array of 2-50 levels; the answer is the expected level index,
  a legend, per-level probabilities, and confidence.

Each question forks the shared state into its own round of prefill, readout, projection,
and softmax. A large question set runs as successive rounds, not as a rejection. Decision
jobs count against the server's concurrency budget, reserve their own execution rows so
chat traffic cannot starve them, and release their slot on every completion path. The
[Decision scoring guide](docs/decisions.md) documents the contract, the execution path,
and the JevBench benchmark (84.4% on 231 public tasks).

https://github.com/user-attachments/assets/aa6b5a2b-a1dd-495c-b6c5-b75b8421ae16

https://github.com/user-attachments/assets/37e6c547-7192-4e2c-b831-2778e216e9dc

### Web UI and llama.cpp compatibility

`--webui` serves a bundled [llama-ui](https://github.com/ggml-org/llama-ui) (llama.cpp /
llama-swap) interface at `/`. The server implements the llama-ui management surface as
real, working functionality backed by serving-owned state in `src/serve/`:

- model load and unload through the router (`POST /models/load`, `POST /models/unload`);
- a router-hooked model-status feed (`GET /models/sse`) that fans out live `model_status`
  events with 5-second keepalives;
- live concurrency slot state (`GET /slots`), one object per configured slot;
- a server-side conversation stream registry keyed by the `X-Conversation-Id` header, with
  replay from a byte offset and a live tail (`POST /v1/streams/lookup`, `GET /v1/stream/{id}`,
  `DELETE /v1/stream/{id}`);
- a server-side tool registry (`GET /tools`, `POST /tools`) that starts empty because NInfer
  is a client-side function-calling engine; and
- owner-addressable cancellation of one in-flight generation
  (`POST /v1/chat/completions/control`), which signals the matching stream's cancel token
  without touching any other request.

These routes are a compatibility surface for the webui, not part of the OpenAI or Anthropic
protocol contract. See [HTTP serving](docs/serving.md).

<img width="1883" height="2114" alt="image" src="https://github.com/user-attachments/assets/90ada720-6658-45c3-a369-a7b83e916683" />

### Copilot and Claude Code tool calls

Serve accepts Copilot custom tools and continues a trailing assistant prefill: a final
text-only assistant message continues its own text instead of opening a new turn. It
parses Claude Code XML tool calls with tolerant truncation recovery — a call whose closing
tags are cut off at the region end still yields a structured result, and the parse
diagnostics report the discarded tail, bounded to a short markup snippet for transparency.
It treats `strict:true` and `required`/named tool choices as advisory: the engine cannot
force a call, so the request proceeds without that guarantee. Tool names run up to 256
bytes, past the 64-byte bound OpenAI documents, so VS Code Copilot's MCP-wrapped names
(`activate_fallback_mcp_<server>_<tool>`) parse cleanly. Engine model metadata enters the
llama.cpp `/models` payload. See [HTTP serving](docs/serving.md).

### Nix packaging

The [flake](flake.nix) builds and packages the `ninfer` and `ninfer-serve` binaries on
CUDA 13.2 and ships a devShell. It pulls httplib, nlohmann_json, spdlog (static), and
utf8proc from nixpkgs instead of vendoring them. It bundles the prebuilt llama-ui Web UI,
pinned to a specific build rather than a rolling `latest` pointer, so `--webui` serves it
from `<exe-dir>/../share/ninfer/webui` with no runtime download. The build clears the CMake
`CMAKE_CXX_SCANDEP_SOURCE` variable so the Ninja generator drops the clang-scan-deps `.ddi`
rule that fails inside the Nix sandbox.

### Engine and performance

The fork also changes engine internals for single-GPU throughput. The additions:

- Short Temporal requests scheduled against a donor work budget, with a Program proof
  required for Persistent backfill.
- A 4/8-warp MMA fast path for INT8 prompt attention, with prefill chunks rounded to
  prompt waves.
- Measured CUDA Graph memory in the engine's memory summary.

---

> Selected checkpoints. Maximum single-GPU inference performance.

NInfer is a from-scratch C++/CUDA inference engine for Qwen3.5 Dense and MoE architectures on a
single NVIDIA GeForce RTX 5090. It runs text, image, and video prompts through a local CLI or
OpenAI-/Anthropic-compatible HTTP APIs. The runtime is deliberately specialized: one GPU, one
resident model, and a startup-fixed capacity of one to eight active requests.

Five official artifacts are available. The quick-start commands use Qwen3.8-27B NVFP4.

| Model | Weights | Artifact | Download and model card |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

Each v3 `.ninfer` artifact carries model configuration, encoded weights, logical bindings and
frontend resources. Runtime execution uses those facts with the implemented model and Op
capabilities. You can also [convert your own weights](docs/weight-conversion.md), reuse an official
recipe or choose another supported mixture of formats.

The current engine requires v3 artifacts. Existing official v2 downloads can be
[upgraded locally](docs/weight-conversion.md#upgrade-an-existing-v2-artifact) without downloading
the weights again.

## Quick start

NInfer requires 64-bit Linux, an NVIDIA GeForce RTX 5090, a CUDA toolkit supporting `sm_120a`,
CMake 3.28 or newer, a C++20 host compiler, Ninja, `pkg-config`, FFmpeg development libraries
(`libavformat`, `libavcodec`, `libavutil`, and `libswscale`), and `libcurl >= 7.85`.
CUDA 13.1 is the validated development toolkit; CMake does not impose a CUDA version floor.
The build rejects CUDA architectures other than `sm_120a`.

Build the product binaries:

```bash
git clone https://github.com/Neroued/ninfer.git
cd ninfer

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Tests and benchmarks are excluded from the default build. `cmake --preset release` configures
the same product build; `cmake --preset dev` also enables tests and benchmarks and finds a
Python 3 interpreter. Both presets use `build/` and explicitly reset the build options.
Machine-specific compiler and Python paths belong in the ignored `CMakeUserPresets.json`.
See [build organization and configuration](docs/maintainer/build-system.md) for details.

There is no install target or packaged binary distribution; run NInfer from its source build tree.
Python tools run independently of CMake; the standalone HBM probe has its own
[build command](tools/README.md#standalone-hbm-probe).

Download the artifact used by this example with the Hugging Face CLI:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

Start a long-running text/agent server with two active-request lanes and explicit Device/Host
checkpoint capacity:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

Each request has a 240,000-token logical ceiling. A shared 240,000-token Device KV pool serves
admitted requests; two requests run concurrently when their combined reservations fit. The cache
tiers provide two Device checkpoint slots, eight pinned Host State slots, and 8 GiB of pinned Host
KV beyond the two active StateImages.

Send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

The `POST /v1/decisions` route (alias `POST /v1/systemone`) runs decision scoring on each
model's loaded engine (no separate
model load); see the [Decision scoring section](docs/serving.md#decision-scoring-post-v1decisions)
of the serving guide.

Run a one-shot CLI request with a 32,768-token allocation:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Answer content is written to stdout. Human-readable startup/runtime diagnostics and the CLI-owned
reasoning, timing, throughput, memory, and speculative-decoding report are written to stderr;
reasoning and the result report remain unprefixed product output. On a terminal, weight
materialization uses one transient progress line followed by a compact Engine-ready summary.
Redirected stderr receives persistent readable progress without terminal control sequences. Use
`--log-level debug` for complete startup detail. Option and local input errors remain direct command
diagnostics. Use `--messages FILE` and `--vision` for structured image/video input; see the
[CLI guide](docs/cli.md) and [committed examples](examples/cli/).

## Resource-aware long-context reuse

A reusable prefix checkpoint contains KV and the complete continuation state for its exact prompt
frontier. A Device-resident checkpoint resumes directly. Under pressure, the planner weighs Device
retention, pinned Host State/KV, and eviction by immediate restore work and later reuse cost. Active
requests retain their completion reservations.

See [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
for the algorithm and [Serve TTFT benchmark](tools/bench/ttft/) for public-HTTP coverage of hot
reuse, Host resume, eviction, shared prefixes, scheduling boundaries, and multimodal load.

## Performance

Published measurements use an RTX 5090. The [performance index](docs/performance.md) links to
per-model run records and the [measurement rules](docs/performance/methodology.md). The tables
below are excerpts from those detailed results. Qwen3.8 uses FP8 E4M3 row-256 KV;
Qwen3.6 uses INT8 group-64 KV.

### Concurrent MTP3 decode

Saturated decode used CUDA Graphs, MTP3, and one 8,192-token generation per active
request. Throughput uses aggregate committed decode tokens from complete intervals whose actual
decode batch equaled the configured concurrency. Acceptance covers the complete request wave;
these rates are steady decode (tok/s).

| Model profile | C=1 tok/s / accept | C=2 tok/s / accept | C=4 tok/s / accept | C=8 tok/s / accept |
|---|---:|---:|---:|---:|
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% |
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#decode-saturation) `groupwise-int` | 642.5 / 68.6% | 907.2 / 66.3% | 1,213.5 / 69.6% | 1,380.7 / 68.0% |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `groupwise-int` | 136.5 / 44.4% | 253.3 / 45.2% | 398.1 / 46.1% | 582.4 / 46.4% |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `nvfp4` | 147.7 / 46.2% | 291.0 / 48.7% | 522.2 / 45.8% | 922.4 / 46.1% |

### Single-request serving

The serial serving corpus used CUDA Graphs, a 1,024-token prefill chunk, and five
fixed seeds after warm-up. The table keeps one short-prefill, one extreme-prefill, and one
structured-output MTP3 point for each published profile; the full context and scenario matrices are
linked from each model below.

| Model profile | 7,680-token prefill | 260,096-token prefill | Structured MTP3 decode |
|---|---:|---:|---:|
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) `groupwise-int` | 17,705.4 tok/s | 5,247.0 tok/s | 779.6 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `groupwise-int` | 3,218.1 tok/s | 1,614.8 tok/s | 193.0 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `nvfp4` | 11,191.5 tok/s | 2,510.6 tok/s | 252.2 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `groupwise-int` | 3,331.9 tok/s | 2,139.4 tok/s | 214.7 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `nvfp4` | 12,819.1 tok/s | 4,016.4 tok/s | 231.7 tok/s |

## Evaluation

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8 rows used temperature
1.0 and presence penalty 0.0. Multimodal evaluation used `--vision` and an 81,920-token context
limit. Text evaluation used 262,144 tokens except Qwen3.8-27B NVFP4, which used 252,928 tokens to
fit the RTX 5090 after weights. Each score is one sample per problem; model cards contain the
correct/total counts and evaluation notes.

## Startup notes

GPU residency is fixed at process startup. `--spec` selects speculative decoding residency, and
`--vision` independently selects Vision residency. Qwen3.6-35B-A3B DFlash can be combined with
Vision; it accelerates generated-text decode after multimodal prefill, not Vision encode itself.

## Docker

Build the runtime image on a host with the NVIDIA Container Toolkit:

```bash
docker build --tag ninfer:local .
```

Mount the downloaded model and run the same example server profile:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

## Capabilities and limits

The official artifacts provide the following capabilities, with optional components enabled at startup:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill, exact-batch CUDA Graph decode, and startup-bounded batched decode;
- MTP speculative decoding with draft windows from one to five;
- BF16, INT8, FP8, NVFP4, and K8V4 KV storage;
- offline causal-perplexity scoring;
- offline decision scoring (serving-only, POST /v1/decisions) with per-option isolated v1
  semantics; the route runs on the model's loaded engine (no separate model load); choice
  questions are capped at the artifact's compiled label-table size (<=255, above it is a 422);
- private and shared exact-prefix reuse with Device/Host State and KV retention;
- model-aware sampling defaults and explicit sampler overrides;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming,
  tools, local response state, token counting, and usage accounting.

The 35B-A3B target additionally supports DFlash with draft windows from one to fifteen for Text and
image/video Vision prompts. Qwen3.8-27B artifacts with the DFlash2 companion weights support
`--spec dflash2 --draft-tokens 7` for the same Text/Vision Engine path, with draft counts 1..15
and either full or optimized proposal heads.

The product boundary remains intentionally small:

- one RTX 5090 and one resident model per Engine;
- a startup-fixed capacity of one to eight active requests with bounded FIFO ingress;
- no request preemption, priority/QoS, active-request swapping, weight offload, multi-GPU, or
  distributed serving;
- one shared startup-fixed KV pool across active requests and retained prefixes;
- model architectures and format/shape combinations use explicitly implemented native paths;
- parsed tool calls are returned to the client; NInfer does not execute tools;
- the in-tree C++ headers are not distributed as an installed SDK.

`--max-context` is each sequence's logical limit. `--kv-capacity` sizes the shared Main Text KV pool
used by active requests and retained prefixes; `auto` resolves the largest legal capacity at
startup from the memory remaining after weights while keeping 1 GiB of sizing headroom. Explicit
capacities remain fixed for the process lifetime.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Weight conversion and custom recipes](docs/weight-conversion.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT benchmark](tools/bench/ttft/)
- [CLI examples](examples/cli/)
- [Contributing](CONTRIBUTING.md)

Run the relevant `--help` for the exact current option contract.

## Support

NInfer is a personal project that I develop out of interest. If you find it useful and would like
to support its continued development, you can [support the project on Ko-fi](https://ko-fi.com/neroued).

Support is entirely voluntary. It is not a purchase or investment and does not come with financial
returns, promised services or features, or a role in project decisions. The project's direction,
priorities, technical choices, and release schedule remain independently determined by the
maintainer.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.
