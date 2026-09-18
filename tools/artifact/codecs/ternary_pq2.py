"""Exact 2-bit ternary PQ2_0 codes, binary16 block scales, and the rotation auxiliary.

The persistent block is 34 bytes: a little-endian binary16 scale followed by 32
code bytes (two bits per weight, low bits first). A per-tensor rotation auxiliary
mirrors the NVFP4 weight-divisor mechanism: a trailing aligned region carrying the
Hadamard descriptor header and one binary32 sign per input lane.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Sequence

import torch

from ..formats import valid_ternary_pq2_scale_word, valid_ternary_pq2_sign_word
from ..layouts import (
    TERNARY_PQ2_BLOCK_BYTES,
    TERNARY_PQ2_ROTATION_BLOCK_SIZE,
    TERNARY_PQ2_ROTATION_HEADER_BYTES,
    TERNARY_PQ2_SIGN_WORD_BYTES,
    ternary_pq2_geometry,
)
from ._tensor_bytes import (
    Payload,
    _exact_uint8_matrix,
    _payload_length,
    _payload_tensor,
)

#: Rotation transform id for the normalized signed Sylvester Walsh-Hadamard transform.
TERNARY_PQ2_TRANSFORM_SYLVESTER = 0


@dataclass(frozen=True, slots=True)
class TernaryPq2Rotation:
    """The per-tensor rotation auxiliary header fields."""

    block_size: int
    transform: int
    gdn_v_grouped: bool
    inverse: bool


def _rotation_header(block_size: int, transform: int, gdn_v_grouped: bool, inverse: bool) -> bytes:
    """Encode the fixed 16-byte rotation header (little-endian).

    Layout: u32 block_size, u8 transform, u8 gdn_v_grouped, u8 inverse, u8 reserved,
    u32 reserved, u32 reserved.
    """

    if block_size != TERNARY_PQ2_ROTATION_BLOCK_SIZE:
        raise ValueError(
            "ternary PQ2_0 rotation block size must be "
            f"{TERNARY_PQ2_ROTATION_BLOCK_SIZE}"
        )
    if transform != TERNARY_PQ2_TRANSFORM_SYLVESTER:
        raise ValueError("ternary PQ2_0 requires the Sylvester Walsh-Hadamard transform")
    if type(gdn_v_grouped) is not bool or type(inverse) is not bool:
        raise TypeError("ternary PQ2_0 rotation flags must be booleans")
    return struct.pack(
        "<IBBBBII",
        block_size,
        transform,
        1 if gdn_v_grouped else 0,
        1 if inverse else 0,
        0,
        0,
        0,
    )


def _exact_fp16_matrix(tensor: torch.Tensor, shape: tuple[int, int], label: str) -> torch.Tensor:
    if tensor.dtype != torch.float16 or tuple(tensor.shape) != shape:
        raise TypeError(f"{label} must be float16 with shape {shape}")
    return tensor.detach().contiguous().cpu()


def _exact_fp32_vector(tensor: torch.Tensor, length: int, label: str) -> torch.Tensor:
    if tensor.dtype != torch.float32 or tuple(tensor.shape) != (length,):
        raise TypeError(f"{label} must be float32 with shape ({length},)")
    return tensor.detach().contiguous().cpu()


def _validate_scale_words(scales: torch.Tensor) -> None:
    scale_words = scales.view(torch.int16).to(torch.int32) & 0xFFFF
    bad_scale = [not valid_ternary_pq2_scale_word(int(word)) for word in scale_words.flatten().tolist()]
    if any(bad_scale):
        raise ValueError("ternary PQ2_0 scales must be finite binary16 words")


def _validate_sign_words(signs: torch.Tensor) -> None:
    bad_sign = [not valid_ternary_pq2_sign_word(int(word)) for word in signs.view(torch.int32).to(torch.uint32).flatten().tolist()]
    if any(bad_sign):
        raise ValueError("ternary PQ2_0 rotation signs must be exactly +/-1.0")


def _validate_words(scales: torch.Tensor, signs: torch.Tensor) -> None:
    _validate_scale_words(scales)
    _validate_sign_words(signs)


def encode_ternary_pq2_blocks(
    codes: torch.Tensor, scales: torch.Tensor, shape: Sequence[int]
) -> bytes:
    """Encode the interleaved 34-byte block array (no rotation auxiliary).

    ``codes`` has shape ``[N, K/4]`` uint8 and ``scales`` shape ``[N, K/128]``
    float16. Each 128-weight group yields a little-endian binary16 scale
    followed by 32 code bytes.
    """

    geometry = ternary_pq2_geometry("ternary_pq2_0", shape)
    groups = geometry.groups_per_row
    codes8 = _exact_uint8_matrix(
        codes, (geometry.n, geometry.k // 4), "ternary PQ2_0 codes"
    )
    scales16 = _exact_fp16_matrix(scales, (geometry.n, groups), "ternary PQ2_0 scales")
    _validate_scale_words(scales16)

    scale_words = scales16.view(torch.int16)
    # Little-endian: the low byte of each binary16 word comes first.
    scale_bytes = scale_words.contiguous().view(torch.uint8).reshape(geometry.n, groups, 2)
    code_blocks = codes8.reshape(geometry.n, groups, 32)
    block = torch.zeros((geometry.n, groups, TERNARY_PQ2_BLOCK_BYTES), dtype=torch.uint8)
    block[:, :, 0:2] = scale_bytes
    block[:, :, 2 : 2 + 32] = code_blocks
    return block.reshape(-1).numpy().tobytes()


def encode_ternary_pq2(
    codes: torch.Tensor,
    scales: torch.Tensor,
    signs: torch.Tensor,
    shape: Sequence[int],
    *,
    gdn_v_grouped: bool = False,
    inverse: bool = False,
) -> bytes:
    """Encode exact ternary PQ2_0 words into the interleaved 34-byte block layout.

    ``codes`` has shape ``[N, K/4]`` uint8 (two bits per weight, low bits first).
    ``scales`` has shape ``[N, K/128]`` float16. ``signs`` has shape ``[K]``
    float32 with every entry exactly +/-1.0.
    """

    geometry = ternary_pq2_geometry("ternary_pq2_0", shape)
    signs32 = _exact_fp32_vector(signs, geometry.k, "ternary PQ2_0 rotation signs")
    _validate_sign_words(signs32)
    payload = bytearray(geometry.payload_bytes)
    block_array = encode_ternary_pq2_blocks(codes, scales, shape)
    payload[: geometry.block_array_bytes] = block_array
    rotation = _rotation_header(
        TERNARY_PQ2_ROTATION_BLOCK_SIZE,
        TERNARY_PQ2_TRANSFORM_SYLVESTER,
        gdn_v_grouped,
        inverse,
    )
    rotation += signs32.numpy().tobytes()
    payload[
        geometry.rotation_offset : geometry.rotation_offset + geometry.rotation_bytes
    ] = rotation
    return bytes(payload)


def decode_ternary_pq2_words(
    payload: Payload,
    shape: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TernaryPq2Rotation]:
    """Decode exact codes, binary16 scales, rotation signs, and the rotation header."""

    geometry = ternary_pq2_geometry("ternary_pq2_0", shape)
    if _payload_length(payload) != geometry.payload_bytes:
        raise ValueError(
            f"ternary PQ2_0 payload has {_payload_length(payload)} bytes, "
            f"expected {geometry.payload_bytes}"
        )
    raw = _payload_tensor(payload, torch.device("cpu"))
    groups = geometry.groups_per_row
    block = raw[: geometry.block_array_bytes].reshape(geometry.n, groups, TERNARY_PQ2_BLOCK_BYTES)
    scale_words = block[:, :, 0:2].contiguous().view(torch.int16)
    scales = scale_words.view(torch.float16).reshape(geometry.n, groups).clone()
    codes = block[:, :, 2 : 2 + 32].reshape(geometry.n, geometry.k // 4).clone()

    rotation = raw[
        geometry.rotation_offset : geometry.rotation_offset + geometry.rotation_bytes
    ]
    header = rotation[: TERNARY_PQ2_ROTATION_HEADER_BYTES].numpy().tobytes()
    block_size, transform, gdn_flag, inverse_flag = struct.unpack_from(
        "<IBBB", header, 0
    )
    if block_size != TERNARY_PQ2_ROTATION_BLOCK_SIZE:
        raise ValueError(
            f"ternary PQ2_0 rotation block size is {block_size}, "
            f"expected {TERNARY_PQ2_ROTATION_BLOCK_SIZE}"
        )
    if transform != TERNARY_PQ2_TRANSFORM_SYLVESTER:
        raise ValueError(f"ternary PQ2_0 unsupported rotation transform {transform}")
    signs = (
        rotation[TERNARY_PQ2_ROTATION_HEADER_BYTES :]
        .contiguous()
        .view(torch.int32)
        .view(torch.float32)
        .reshape(geometry.k)
        .clone()
    )
    _validate_words(scales, signs)
    return (
        codes,
        scales,
        signs,
        TernaryPq2Rotation(
            block_size=block_size,
            transform=transform,
            gdn_v_grouped=bool(gdn_flag),
            inverse=bool(inverse_flag),
        ),
    )


def dequantize_ternary_pq2(
    payload: Payload,
    shape: Sequence[int],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct the logical ``[N,K]`` matrix from exact stored words (test helper).

    Each represented weight is ``(codebook[code] - 1) * scale`` where the codebook is
    ``{00: -1, 01: 0, 10: +1, 11: +2}`` and ``code`` is the two-bit word.
    """

    codes, scales, _, _ = decode_ternary_pq2_words(payload, shape)
    codebook = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    n, code_bytes = codes.shape
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.int32)
    bits = (codes.to(torch.int32).unsqueeze(-1) >> shifts) & 0x3  # [N, K/4, 4]
    bits = bits.reshape(n, code_bytes * 4)  # natural k order: byte-major, lane-minor
    scale = scales.float().unsqueeze(-1).repeat_interleave(128, dim=-1)
    values = codebook[bits] * scale
    return values.to(dtype)
