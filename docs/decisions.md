# Decision scoring

Decision scoring answers a set of questions over a shared state **without generating tokens**:
each question is scored against the loaded model and a per-candidate probability distribution is
returned. `output_tokens` is always `0`. The route runs on the model the router has loaded (no
separate model load) and is serialized against in-flight generation rounds by the engine's
execution lock.

The model's execution path (state prefill, branch fork, suffix prefill, readout, projection,
candidate softmax) is in [Decision scoring in the model reference](maintainer/qwen3_5-model.md#decision-scoring);
the binding design choices are below.

## Endpoint

`POST /v1/decisions`. The request names **no model** (a top-level `model` field is a 422
unknown field): the route runs on the resident model; with nothing loaded it answers 503
`model_not_ready`. The response's `model` field echoes the loaded model id.

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

A `messages` request supplies the context as chat history, with optional image parts (this
example scores a question over an image):

```bash
curl http://127.0.0.1:8080/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}},
        {"type": "text", "text": "What color is the bicycle in the image?"}
      ]}
    ],
    "questions": {
      "color": {
        "type": "choice",
        "instructions": "What color is the bicycle?",
        "criteria": {"red": null, "blue": null}
      }
    }
  }'
```

## Request

The context is supplied by exactly one of `state` or `messages` (a JSON null value is not a
supplied context; supplying both, or neither, is a 422).

`state` is a string (verbatim), a JSON object/array (JSON-dumped UTF-8), or JSON null (rendered
as the text "null").

`options` is an optional diagnostics object. The only supported key is `raw_logits` (a boolean,
default false): when true, each answer carries a `raw_logits` map (option to pre-softmax readout
logit) parallel to `probabilities`. An unknown option field is a 422 (the field is named).

`messages` is a non-empty array of `{role, content}`. `role` is `system`, `developer`, `user`,
or `assistant`. `content` is a string (one text part) or an array of parts, each `{type, ...}`:
`{"type": "text", "text": ...}` or `{"type": "image_url", "image_url": ...}`. An `image_url`
value is a string URL or an object `{url, detail?}`; the URL is an HTTP(S) URL or a base64 data
URI, and `detail` is omitted or `"auto"` (an explicit profile is a 422). Image input requires the
model's `"vision": true` serve-config field; without it the request fails with 400
`vision_disabled`. The image is expanded to Vision tokens in the shared state prefix; the
questions are then scored over that prefix.

`questions` is a map from question id to `{type, instructions, criteria}`, where `type` is
`noul`, `choice`, or `score`, `instructions` is optional (default null, any JSON value), and the
`criteria` rules follow the type:

- `noul`: optional map, keys restricted to `true`/`false` (any-JSON values);
- `choice`: map of 2-255 entries from key to description (any-JSON values), at most the
  artifact's compiled label-table size (<=255) entries, key order preserved;
- `score`: ordered array of 2-50 level descriptions (any-JSON values).

Unknown fields are a 422 (the offending field is named), including a top-level `model` field,
any field in a question object other than `type`, `instructions`, and `criteria`, any field in a
message object other than `role` and `content`, and any `options` key other than `raw_logits`.

## Response

`{model, answers{<id>: {type, ...}}, usage{input_tokens, output_tokens: 0}}` with per-type answer
fields:

- `noul`: the `noul` probability (P(true)) and no confidence field;
- `choice`: the winning `choice`, per-option `probabilities`, and `confidence`;
- `score`: `score` as the expected 0-based level index, a `legend` mapping level indices to their
  descriptions, per-level `probabilities`, and `confidence`.

When the request set `options.raw_logits`, each answer also carries a `raw_logits` map (option to
pre-softmax readout logit) parallel to `probabilities`.

`usage.output_tokens` is always `0`: decision scoring generates no tokens. Errors: 401 for a
missing/invalid API key, 422 for body validation (the offending field is named), and 503 for a
model that is not loaded (or whose load failed). Limits: choice questions take 2-255 options
(capped at the artifact's compiled label-table size, <=255; out of range is a 422) and score
levels are 2-50.

v1 semantics: each option is scored in isolation (a slice softmax over each question's candidate
set). Option-set interaction is limited to the shared slice denominator, so changing the option
set rescales the surviving options' probabilities; probabilities and confidence are uncalibrated
slice statistics (confidence = normalized Gini).

## Execution

A decision job has one shared state prefix and one branch per question. The model executes the
prepared branches in one serialized job (a bounded decision branch workspace, capacity 16 by
default with a per-branch tail cap, is planned alongside the model at startup; the capacity is
set by the server's `--max-decision-branches` flag or the model's `maxDecisionBranches`
serve-config field):

1. **State prefill.** The shared state prefix is prefilled into the Main KV row 0. For an image
   state the prefix is the media-expanded form (Vision tokens plus text); it is prefilled through
   the multimodal path (Vision encode + text), then forked.
2. **Branch fork.** Row 0 forks into one branch row per question, each with its bounded branch KV
   tail.
3. **Suffix prefill.** Each branch's question suffix (the rendered question text plus its
   criteria) is appended in a batched suffix prefill over the branch rows.
4. **Readout.** Each branch's last-position hidden state is gathered into the persistent
   `decision_readout_hidden` buffer (`[hidden, branches]`).
5. **Projection.** The readout hidden states are projected through the full `output_head` into
   the persistent `decision_readout_logits` buffer (`[vocab, branches]`).
6. **Candidate gather and softmax.** Each branch's candidate token logits are gathered from the
   `[vocab, branches]` readout, and a temperature-scaled group-pooled softmax
   (`ops::candidate_slice_softmax`) produces the per-candidate probability slice.
7. **Answer.** Per question type: `noul` reports `P(true)` with no confidence; `choice` reports
   the winning key, per-option probabilities, and a normalized Gini confidence; `score` reports
   the expected 0-based level index, the level legend, per-level probabilities, and a normalized
   Gini confidence.

## Design decisions

**Readout buffers live in the persistent layout, not the scratch arena.** The decision readout
buffers (`decision_readout_hidden` and `decision_readout_logits`) are planned into the program's
persistent layout, sized for the maximum branch capacity and sliced to the active branch rows.
The batched suffix prefill resets the scratch arena to offset 0 for its chunk intermediates; had
the readout buffers lived in that arena at offset 0, the suffix prefill would clobber them before
the last-position gather and the `output_head` projection ran, feeding non-finite (NaN/inf)
values to the projection. `PersistentLayout` carries the two optional readout tensors
(`program/planning/startup.{h,cpp}`), planned only when `decision_scoring` is set; `ProgramImpl`
binds them and `inspect_decision_admission` refuses to run when they are absent.

**`candidate_slice_softmax` broadcasts the block sum to every active thread.** `block_reduce_sum`
returns the full block sum only to lane 0 (the other threads hold a partial sum or zero). The
kernel writes the sum to a shared `sum_result` and re-reads it after a barrier, so every active
candidate divides by the same row sum. Without the broadcast, candidate columns after the first
divided by a partial or zero sum and yielded `inf` (only the first candidate was correct). The
other `block_reduce_sum` callers (`target_logprobs`, `rmsnorm`, `gdn_gating_proj`,
`softmax_attention`) already consume the value from thread 0 into a shared location, so they are
unaffected.

## Verification

Both decisions are pinned by `tests/models/qwen3_5/test_engine_decision.cpp`
(`ninfer_qwen3_5_decision_test`), a real-artifact oracle: every per-candidate probability
must be finite, in `[0,1]`, and each question's probabilities must sum to one. A NaN/inf in any
candidate fails the test and pinpoints the readout defect at the pipeline level.

## Benchmark

Measured against [JevBench v1.2](https://benchmarkheaven.com/jev-models), a decision-model
benchmark. The public set is 231 typed decisions (48 easy, 111 hard, 72 original): each is a
`state` plus a bounded rubric, answered with a typed result and per-option probabilities. The run
drives the native `POST /v1/decisions` route directly (state + question in, typed probabilities
out), not the chat-completions verbalization path.

Setup: model `ukisai/Swift-Qwen3.8-27B`, live server, RTX 5090, three in-flight decisions. All
231 tasks scored, 0 errors, 0 admission blocks. Wall clock 19 s.

### Headline

| Metric                            | Value           |
| --------------------------------- | --------------- |
| Accuracy (231 public)             | 84.4% (195/231) |
| ECE (10 bins)                     | 0.045           |
| Brier                             | 0.081           |
| Latency p50 / p95                 | 127 ms / 699 ms |
| Speed axis (JevBench ×2 + 0.15 s) | ~81.7           |

### Accuracy by tier and type

| Tier     | choice          | noul          | score         | total           |
| -------- | --------------- | ------------- | ------------- | --------------- |
| easy     | 100% (36/36)    | 100% (12/12)  | —             | 100% (48/48)    |
| original | 97% (35/36)     | 95% (23/24)   | 100% (12/12)  | 97.2% (70/72)   |
| hard     | 70% (47/67)     | 73% (28/38)   | 33% (2/6)     | 69.4% (77/111)  |
| all      | 84.9% (118/139) | 85.1% (63/74) | 77.8% (14/18) | 84.4% (195/231) |

### Hard tier by family

| Family           | Acc         |
| ---------------- | ----------- |
| temporal-numeric | 33% (5/15)  |
| probability      | 50% (5/10)  |
| tradeoff         | 67% (4/6)   |
| long-policy      | 68% (13/19) |
| multi-hop        | 72% (13/18) |
| judge-hard       | 76% (13/17) |
| ambiguous        | 86% (6/7)   |
| trap             | 88% (7/8)   |
| adversarial      | 100% (6/6)  |
| routing-hard     | 100% (5/5)  |

### Latency (seconds)

| Scope    | p50   | p90   | p95   | p99   | max   | mean  |
| -------- | ----- | ----- | ----- | ----- | ----- | ----- |
| all      | 0.127 | 0.619 | 0.699 | 0.896 | 0.925 | 0.238 |
| easy     | 0.100 | 0.103 | 0.453 | 0.691 | 0.736 | 0.134 |
| original | 0.101 | 0.123 | 0.123 | 0.309 | 0.331 | 0.109 |
| hard     | 0.286 | 0.694 | 0.774 | 0.911 | 0.925 | 0.366 |

Hard-tier p95 (774 ms) is prefill-dominated by the ~3.7k-token states. Tail latency (p95 and
above) is queueing-sensitive: a single decision worker serializes execution, so a task can queue
behind heavier in-flight decisions, inflating per-tier tails (visible in the easy row). p50 and
mean reflect the compute cost; accuracy and calibration are unaffected by queueing.

### Calibration

36 of 231 wrong. ECE 0.045 (10 bins), Brier 0.081 — the model is well calibrated overall. The
five most-confident wrong answers:

| Task                            | Conf  | Predicted            | Gold                    |
| ------------------------------- | ----- | -------------------- | ----------------------- |
| hard-opus-b-tradeoff-01         | 0.949 | escalate_to_security | escalate_to_engineering |
| hard-opus-a-probability-08      | 0.852 | yes                  | no                      |
| hard-sol-b-judge-hard-02        | 0.852 | yes                  | no                      |
| hard-opus-c-temporal-numeric-02 | 0.798 | no                   | yes                     |
| hard-opus-a-probability-04      | 0.729 | late                 | on_time                 |

### Field comparison (same 231 public tasks)

Every published system's public-set accuracy, against its published JevBench composite (the
composite is a geometric mean over Intelligence, Calibration, Speed, and Cost on the full 534-task
set, so it is not directly comparable to a public-set accuracy). This run has no published
composite (self-hosted, no Speed/Cost measurement):

| Rank | System                                         | Score | Pub acc   |
| ---- | ---------------------------------------------- | ----- | --------- |
| 1    | Jev 1.13.0 (TypeSafe AI)                       | 75.4  | 86.6%     |
| 2    | SemIf, formerly OpenJev (Qwen3.5-4B)           | 74.7  | 81.0%     |
| 3    | djev (Maisa, diffusion-gemma)                  | 74.3  | 84.0%     |
| 4    | openJev Verdict 1.4                            | 72.5  | 57.6%     |
| 5    | Laya (ModernBERT-large 421M)                   | 70.1  | 58.4%     |
| 6    | open-alternative-jev (Qwen3.5-4B)              | 69.8  | 74.0%     |
| 7    | system-one-open (Gemma 4 E2B LoRA)             | 68.9  | 73.2%     |
| •    | **NInfer ukisai/Swift-Qwen3.8-27B (this run)** | **68.4** | **84.4%** |
| 8    | OpenJev (DiffusionGemma 26B NVFP4)             | 67.7  | 81.8%     |
| 9    | SimpleJev Qwen3.8-27B                          | 67.3  | 86.6%     |
| 10   | jeff (GLiFormer 400M)                          | 66.9  | 62.8%     |
| 11   | kev 0.6B                                       | 66.7  | 66.7%     |
| 12   | openjev-sglang (Qwen3.6-35B-A3B)               | 66.3  | 85.3%     |
| 13   | openJev Verdict (ModernBERT-base 151M)         | 66.2  | 55.4%     |
| 14   | GPT-5.6 Luna (low reasoning)                   | 66.2  | 97.4%     |
| 15   | open-jev-deberta-v3-large (CPU)                | 64.6  | 52.4%     |
| 16   | SimpleJev Qwen3.6-35B-A3B                      | 63.8  | 81.4%     |
| 17   | Bespoke Nimble 9B                              | 63.7  | 67.5%     |
| 18   | kev 0.5B                                       | 63.1  | 49.4%     |
| 19   | kev 4B                                         | 62.2  | 66.2%     |
| 20   | Gemini 3.1 Flash-Lite                          | 60.9  | 87.0%     |
| 21   | kev 8B                                         | 58.3  | 71.4%     |
| 22   | DeepSeek V4.1 Flash (thinking)                 | 57.8  | 97.8%     |
| 23   | system-one (Qwen3-8B)                          | 56.6  | 71.9%     |
| 24   | GLiNER2 (Fastino)                              | 53.0  | 58.0%     |
| —    | classifier.dev (fast tier)                     | 84.8  | 85.3%     |
| —    | Qwen3.8 27B (Chutes TEE)                       | 25.5  | 71.9%     |
| —    | Needle 3 (options as tools)                    | 19.2  | 22.1%     |
| —    | Needle 3 (Cactus, 2-bit, CPU)                  | 16.8  | 22.5%     |

On public-set accuracy the run is mid-pack: below the LLM baselines (DeepSeek 97.8%, GPT-5.6
Luna 97.4%, Gemini 87.0%) and the top Jev systems (Jev 86.6%, SimpleJev Qwen3.8-27B 86.6%),
around openjev-sglang and classifier.dev (85.3%) and djev (84.0%), and above every 4B-or-smaller
open rebuild.

• This run is not a published JevBench system; its row is inserted at its Score position (8th by
Score, between system-one-open 68.9 and OpenJev 67.7). The 68.4 is a public-set estimate computed with JevBench's
exact axis formulas (`score_ninfer_run.py`): Intelligence 89.3, Calibration 90.9 (ECE-only upper
bound — the probability-fidelity component lives in the held-out set), Speed 82.0, Cost 32.8
(hosted-provider estimate at the OpenRouter Qwen3.8-27B list price). The full published Score also
uses the 109 held-out hard items, so the true Score is somewhat below this estimate. Self-hosted at
~$0 per decision the composite is 90.3 — higher, but not comparable to the table, which prices
self-hosted systems at a hosted-provider rate.
