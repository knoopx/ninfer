# Ternary PQ2_0 model reference

This reference specifies native PQ2_0 support in NInfer for the `prism-ml/Ternary-Bonsai-2-27B`
model. It is the P0 design document: format, rotation contract, source package, sm_120a kernel
plan, change surface, verification, and naming. It does not implement P1-P4.

The model is Qwen3.8-27B. NInfer already implements that architecture as `qwen35`. The new
element is the weight representation: ternary codes in a Hadamard-rotated basis, with the rotation
applied to activations at runtime.

[Numeric formats](tensor-formats.md) owns the persistent format registry;
[Qwen3.5 model](qwen3_5-model.md) owns the architecture;
[Op development](op-development.md) owns the qualification contract used in Section 8.

Source materials for the facts in this document:

- Prism llama.cpp fork (branch `prism`): `ggml-common.h` block definition, `ggml-quants.c`
  reference codec, `ggml-cuda/fwht.cu` activation transform, `conversion/base.py` Hadamard
  manifest contract;
- `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` package: `model.safetensors`, `hadamard.json`,
  `runtime/codec.py`;
- white paper quality anchors (MMLU-Redux 89.09, GSM8K 96.66).

## 1. Model and format overview

The language model is Qwen3.8-27B: 64 Text layers (16 full attention, 48 GDN), hidden width 5120,
vocabulary 248320, untied embedding and output head. This matches the official 27B Dense geometry
in the [Qwen3.5 model reference](qwen3_5-model.md). The package carries no MTP component. A
vision tower is present in the package but is out of scope: the NInfer artifact binds the Text
component only.

The weight scheme is `TERNARY_PQ2_0` (proposed registry short name `pq2`):

| Property | Value |
|---|---|
| Codes | 2 bits/weight, codebook `{−1, 0, +1, +2}` |
| Scale | one IEEE-754 binary16 multiplier per group of 128 weights |
| Logical cost | 2.125 bits/weight |
| Block size | 34 bytes per 128 weights |

All 401 folded matrices (attn and MLP projections, output head) use the ternary
scheme. The embedding uses the same scheme on the inverse-transform route. Full-precision
tensors cover normalization, GDN gating, and small recurrent tensors:
26,238,464 parameters, 0.0976% of the language model (Section 5).

Every folded matrix is stored rotated. The converter folds the Hadamard rotation into the stored
weights (Section 3). The runtime applies the rotation to activations before each folded matmul,
fused into the GEMM A-tile load. There is no separate memory pass and no dequantization path.

## 2. PQ2_0 byte-level specification

A PQ2_0 block holds 128 weights in 34 bytes:

```text
bytes 0..1   FP16 little-endian scale d
bytes 2..33  32 code bytes, 2 bits per weight
```

Weight `j` of the 128 sits in code byte `j/4`, bits `2*(j%4)` through `2*(j%4)+1`, low bits
first. The codebook is:

```text
00  ->  -1
01  ->   0
10  ->  +1
11  ->  +2
```

The represented weight is `(q - 1) * d`. `q` is the 2-bit code and `d` is the exact binary16
scale expanded to the working precision.

The reference quantizer, per 128-group:

```text
d = fp16(max_i |x[i]|)
q[i] = clamp(round(x[i] / d) + 1, 0, 3)
```

An all-zero group emits `d = +0` and code `01` everywhere, reconstructing zero. A stored scale
must be finite; the MLX producer rejects non-finite scales.

### 2.1 Worked example

One 128-group with `d = 0.5` (FP16 bit pattern `0x3800`, stored little-endian as `00 38`):

```text
weights w0..w3 = -0.5, 0.0, +0.5, +0.25
```

- `w0`: `round(-0.5/0.5) + 1 = 0` -> code `00`
- `w1`: `round(0.0/0.5) + 1 = 1` -> code `01`
- `w2`: `round(+0.5/0.5) + 1 = 2` -> code `10`
- `w3`: `round(0.25/0.5) + 1 = round(0.5) + 1 = 1` -> code `01` (ties to even)

Weights 4..127 are zero: code `01` each. The block is:

```text
bytes 0..1   00 38
byte 2       64      = 0b01_10_01_00  (w3 w2 w1 w0)
bytes 3..33  55 x 31 = 0b01010101     (code 01 per weight)
```

Decode of byte `0x64`: `w0 = (0-1)*0.5 = -0.5`, `w1 = 0`, `w2 = +0.5`, `w3 = 0`. The reconstructed
`w3` is 0 against a source of 0.25; that is the scheme's quantization error, not a decode error.

## 3. Rotation contract

The rotation is the normalized signed Sylvester Walsh-Hadamard transform with block size 1024:

```text
R = (1 / sqrt(1024)) * H_1024 * S
```

`H_1024` is the Sylvester Hadamard matrix, `S` is a diagonal sign table fixed per input width.
The fold stores `W_rot` on the input-last-dimension axis. The runtime recovers the original
product by transforming the activation:

```text
A(x) = (1 / 1024^0.5) * FWT(x * s)
```

`x * s` applies the per-width sign vector elementwise before the fast Walsh-Hadamard transform.
The fork kernel fuses both the sign flip and the `1/sqrt(N)` scale into the load.

The transform is orthogonal, so the inverse is the same kernel with the sign vector:
`R^-1 = (1/sqrt(1024)) * H_1024^T * S` and `H_1024` is symmetric. The embedding lookup uses the
inverse transform after the row gather, marked by `inverse_weight_names`.

### 3.1 Manifest keys

`hadamard.json` is the version-1 Prism Hadamard contract:

| Key | Value for this model |
|---|---|
| `prism.hadamard.version` | 1 |
| `prism.hadamard.block_size` | 1024 |
| `prism.hadamard.transform` | `normalized-sylvester-walsh-hadamard` |
| `prism.hadamard.axis` | `input-last-dimension` |
| `prism.hadamard.sign_mode` | `explicit` |
| `prism.hadamard.weight_names` | 401 folded tensors |
| `prism.hadamard.inverse_weight_names` | 1 entry: `language_model.model.embed_tokens.weight` |
| `prism.hadamard.sign_widths` | `[5120, 6144, 17408]` |
| `prism.hadamard.sign_values` | 28672 values, each `+1` or `-1` |
| `prism.hadamard.gdn_v_grouped` | `true` |

The sign table stores one vector per distinct input width, 28672 values total. All three widths
are multiples of the 1024 block size. The converter validates the contract before encoding:
block size a power of two, every sign width positive and a multiple of the block size, every
sign vector the right length with only `+1`/`-1`, and every folded tensor on a verified
matmul path.

### 3.2 GDN grouped-V order

`gdn_v_grouped` records that the GDN `out_proj` matrix keeps its grouped-V row order. The GDN
state already feeds grouped-V activations to that matmul. A consumer must not re-permute the
GDN activations for this model.

## 4. Source package specification

The converter reads the MLX package directly: `model.safetensors` plus `hadamard.json`. It never
parses GGUF.

### 4.1 Ternary matrices

The package stores 402 ternary matrices: the 401 folded tensors plus the inverse-transform
embedding. Each matrix `X` carries four tensors:

| Tensor | Dtype | Shape | Role |
|---|---|---|---|
| `X.weight` | U32 | `[rows, width/16]` | ternary codes |
| `X.scales` | F16 | `[rows, width/128]` | one FP16 scale per 128-group |
| `X.biases` | F16 | `[rows, width/128]` | exactly `-scales`, redundant, dropped |
| `X.signs` | F32 | `[width]` | per-matrix sign vector |

A U32 word packs 16 lanes of 2 bits. Lane `l` owns bits `2l` through `2l+1`. The 16 words of a
128-group equal the 32 GGUF code bytes reinterpreted as little-endian U32. The NInfer encoder
therefore performs a pure byte repack with zero code transcoding:

```text
NInfer block (128 weights)
  bytes 0..1   <- X.scales group, FP16 little-endian
  bytes 2..33  <- 8 U32 words of X.weight, little-endian byte order
```

Weight `j` of the group then decodes through the Section 2 layout. No dequantization path is
introduced: the runtime decodes 2-bit slots in-kernel.

### 4.2 Full-precision and dropped tensors

The package stores 2390 model tensors. Section 5 lists the full-precision set. The vision tower
F16 tensors and `X.biases` are dropped by the recipe.

## 5. Full-precision tensor table

The language model stores 851 F32 tensors. Of those, 402 are the per-matrix `signs` vectors
(2,910,208 floats). The remaining 449 hold 26,238,464 full-precision parameters, 0.0976% of the
language model:

| Tensor pattern (layer `i`) | Shape |
|---|---|
| `layers/i.input_layernorm.weight` | `[5120]` |
| `layers/i.post_attention_layernorm.weight` | `[5120]` |
| `layers/i.linear_attn.norm.weight` | `[128]` (GDN layers) |
| `layers/i.linear_attn.A_log` | `[48]` |
| `layers/i.linear_attn.dt_bias` | `[48]` |
| `layers/i.linear_attn.conv1d.weight` | `[10240, 4, 1]` |
| `layers/i.linear_attn.in_proj_a.weight` | `[48, 5120]` |
| `layers/i.linear_attn.in_proj_b.weight` | `[48, 5120]` |
| `layers/i.self_attn.q_norm.weight` | `[256]` (full-attention layers) |
| `layers/i.self_attn.k_norm.weight` | `[256]` |
| `model.norm.weight` | `[5120]` |

The GDN `a`/`b` projections (`in_proj_a`, `in_proj_b`) stay full precision. They feed the GDN
recurrence and are small. The vision tower and MTP components do not bind into the artifact.

## 6. sm_120a kernel plan

The target is consumer Blackwell `sm_120a`. It has no TMEM or tcgen05 path: GEMM uses the
classic `mma.sync m16n8k16` tensor-core path with BF16/FP16 inputs and FP32 accumulation. TMA
is available for tile loads. NInfer already ships NVFP4 and FP8 sm_120a kernels under
`src/ops/linear/`. The ternary family mirrors the NVFP4 pattern. It fuses packed codes and
group scale into the MMA. A 2-bit slot unpack yields `{−1, 0, +1, +2}`. The FWT applies to
the A tile.

### 6.1 Activation FWT

The fork `fwht.cu` is the reference. It keeps three launch shapes:

- a warp-register kernel for widths 64/128/256, four rows per block;
- a block kernel for widths 512 through 8192, 256 threads per row, register staging with
  shared-memory butterfly stages;
- the sign flip and the `1/sqrt(N)` scale fused into the load.

The butterfly convention is fixed: the low element of a pair takes `x + y`, the high element
takes `x - y`. Block 1024 uses the block kernel.

For large token extents, a tensor-core FWT is the reference: HadaCore computes the Walsh-Hadamard
transform with `mma.sync m16n8k16` in 16x16 chunks across sizes 2^2 through 2^15. The SM120
adaptation (VLLM-TurboQuant-SM120) runs those chunks in BF16 and converts each 16x16 chunk back
with `cvt.rn.bf16x2`. P1 must pin that chunk-rounding semantics: FP32 accumulation within a
16x16 MMA chunk, round-to-nearest BF16 conversion at chunk boundaries. The FP64 oracle of
Section 8 qualifies both the exact and the chunked variants against that pinned rule.

The TurboQuant SM120 result picks the dispatch: a shared-memory butterfly kernel serves
small-batch decode, the MMA path serves large token extents.

### 6.2 Ternary GEMM family

Each consumer op gains a `ternary` format directory next to its `nvfp4` one:

```text
src/ops/linear/ternary/
src/ops/linear_swiglu/ternary/
src/ops/gdn_input_proj/ternary/
src/ops/attn_input_proj/ternary/
src/ops/linear_add/ternary/
src/ops/linear_pair/ternary/
```

- A tile: TMA loads the activation tile, applies `x * s` then the FWT block, and converts to BF16.
  The transform is fused into the load; it writes no intermediate.
- B tile: the 32 code bytes of each 128-group unpack to 128 slots of `{−1, 0, +1, +2}`. The FP16
  group scale multiplies the slots, mirroring the NVFP4 block-scale fold into the MMA.
- Decode shape (batch one): a GEMV variant vectorizes the slot unpack and the transform.

The per-matrix sign vector is a rotation resource. It is stored as an auxiliary on the
mathematical input, analogous to the NVFP4 site-level input divisor, not as part of the numeric
format.

The embedding route gathers rows first, then applies the inverse transform per
`inverse_weight_names`. No other table uses the inverse path.

## 7. NInfer change surface

The P1-P4 work touches these locations. This section lists them; it implements nothing.

| Location | Change |
|---|---|
| `tools/artifact/formats.py` | register `TERNARY_PQ2_0` (short name `pq2`) in the closed `NumericFormat` registry |
| `src/core/weight.h` | `QType::TERNARY_PQ2_0`; a block storage layout entry |
| `src/artifact/formats.cpp` | `kFormats` and `kLayouts` registry entries |
| `tools/convert/methods.py` | encoded-source method that preserves U32 code words, FP16 scales, and F32 signs, with the sign vector exposed as an auxiliary |
| `tools/convert/official_recipes.py` | `ternary_bonsai_2_27b` recipe, patterned on `qwen3_8_27b_nvfp4`: assign every projection to `pq2`, keep the Section 5 set direct |
| `src/ops/{linear,linear_swiglu,gdn_input_proj,attn_input_proj,linear_add,linear_pair}/` | new `ternary/` format directories per Section 6.2 |
| embedding op | inverse-rotated `embed_tokens` route per Section 6.2 |

The converter reads `model.safetensors` and `hadamard.json`. It validates the Hadamard contract
of Section 3.1 and the Section 4.1 tensor set before encoding. It drops `X.biases` and the
vision tower.

## 8. Verification plan

The qualification contract is the one in [Op development](op-development.md). The independent
oracle is FP64.

- Codec: exact trit decode. Random and boundary 34-byte blocks (zero scale, subnormal FP16
  scales, all four codes per slot) decode to the FP64 reference values.
- FWT: exact Walsh-Hadamard in FP64 at width 1024, including the sign convention and the
  `1/sqrt(1024)` normalization. The BF16 chunked variant is qualified against the pinned
  Section 6.1 chunk-rounding rule.
- Byte repack: the MLX U32 plus FP16 scale repack of Section 4.1 must decode identically in both
  directions with zero code change.
- Bias drop: the converter checks `X.biases == -X.scales` exactly before dropping the tensor.
  A mismatch fails the conversion.
- Ops: real shapes from Section 5 and the projection set, against the naive FP32/FP64 matmul
  oracle over the exact-decoded weights.
- End-to-end: perplexity on the fixed corpus, then the white-paper quality anchors:
  MMLU-Redux about 89.09 and GSM8K about 96.66.

GPU verification runs on the user's machine. The plan states the commands and criteria; it does
not execute them here.

## 9. Naming and deployment

- Model id: `prism-ml/ternary-bonsai-2-27b`.
- Format: `QType::TERNARY_PQ2_0`, registry short name `pq2`.
- Artifact filename: model slug plus the dominant-format suffix, following the `nvfp4` pattern.
  The artifact is `prism-ml_ternary-bonsai-2-27b_pq2.ninfer`.
- Deploy directory: `~/.local/share/ninfer/models/prism-ml/`.
- The `Model artifacts` table in [docs/README.md](../README.md) gains one row on publication.
