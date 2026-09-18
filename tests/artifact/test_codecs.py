from __future__ import annotations

import struct

import pytest
import torch

from tools.artifact.layouts import (
    block_scale_geometry,
    ternary_pq2_geometry,
)
from tools.artifact.codecs.direct import decode_direct, encode_direct
from tools.artifact.codecs.nvfp4 import decode_nvfp4_words, encode_nvfp4
from tools.artifact.codecs.row_split import decode_row_split_codes, encode_row_split
from tools.artifact.codecs.ternary_pq2 import decode_ternary_pq2_words, encode_ternary_pq2


def _signed_word(word: int, bits: int) -> int:
    return word if word < 1 << (bits - 1) else word - (1 << bits)


@pytest.mark.parametrize(
    ("format_name", "tensor", "words", "word_format", "word_view"),
    [
        (
            "bf16",
            torch.tensor(
                [_signed_word(word, 16) for word in (0x0000, 0x8000, 0x0001, 0x7FC1)],
                dtype=torch.int16,
            ).view(torch.bfloat16),
            (0x0000, 0x8000, 0x0001, 0x7FC1),
            "H",
            torch.int16,
        ),
        (
            "fp32",
            torch.tensor(
                [
                    _signed_word(word, 32)
                    for word in (0x00000000, 0x80000000, 0x00000001, 0x7FC01234)
                ],
                dtype=torch.int32,
            ).view(torch.float32),
            (0x00000000, 0x80000000, 0x00000001, 0x7FC01234),
            "I",
            torch.int32,
        ),
        (
            "int32",
            torch.tensor((0, -1, -(1 << 31), (1 << 31) - 1), dtype=torch.int32),
            (0, -1, -(1 << 31), (1 << 31) - 1),
            "i",
            torch.int32,
        ),
    ],
)
def test_direct_layout_preserves_exact_little_endian_words(
    format_name, tensor, words, word_format, word_view
):
    expected = struct.pack("<" + word_format * len(words), *words)
    payload = encode_direct(tensor, format_name)
    assert payload == expected
    decoded = decode_direct(payload, format_name, tensor.shape)
    assert torch.equal(decoded.view(word_view), tensor.view(word_view))

    if format_name == "bf16":
        with pytest.raises(TypeError):
            encode_direct(tensor.float(), format_name)


@pytest.mark.parametrize(
    (
        "format_name",
        "k",
        "group_size",
        "k_pad",
        "scale_offset",
        "prefix",
        "base_prefix",
        "high_prefix",
    ),
    [
        pytest.param(
            "q4_g64_fp16",
            65,
            64,
            128,
            256,
            (-8, -7, -1, 0, 1, 7),
            b"\x98\x0f\x71",
            b"",
            id="q4",
        ),
        pytest.param(
            "q5_g64_fp16",
            130,
            64,
            256,
            512,
            (-16, -15, -1, 0, 1, 15),
            b"\x10\x0f\xf1",
            b"\x07",
            id="q5",
        ),
        pytest.param(
            "q6_g64_fp16",
            65,
            64,
            128,
            512,
            (-32, -31, -17, -16, -1, 0, 15, 31),
            b"\x10\x0f\x0f\xff",
            b"\xea\x43",
            id="q6",
        ),
        pytest.param(
            "q8_g32_fp16",
            33,
            32,
            128,
            256,
            (-127, -1, 0, 1, 127),
            b"\x81\xff\x00\x01\x7f",
            b"",
            id="q8",
        ),
    ],
)
def test_row_split_matches_known_packed_bytes(
    format_name, k, group_size, k_pad, scale_offset, prefix, base_prefix, high_prefix
):
    groups = k_pad // group_size
    codes = torch.zeros((1, groups, group_size), dtype=torch.int8)
    codes[0, 0, : len(prefix)] = torch.tensor(prefix, dtype=torch.int8)
    scales = torch.zeros((1, groups), dtype=torch.float16)
    scales[0, :2] = torch.tensor([1.5, 0.25], dtype=torch.float16)

    # Literal wire positions and words come from the format, independently of layout helpers.
    expected = bytearray(scale_offset + groups * 2)
    expected[: len(base_prefix)] = base_prefix
    expected[256 : 256 + len(high_prefix)] = high_prefix
    expected[scale_offset:] = struct.pack(
        "<" + "H" * groups, 0x3E00, 0x3400, *([0] * (groups - 2))
    )
    assert encode_row_split(codes, scales, format_name, (1, k)) == expected
    decoded_scales, decoded_codes = decode_row_split_codes(
        expected, format_name, (1, k)
    )
    assert torch.equal(decoded_scales, scales)
    assert torch.equal(decoded_codes, codes)


def test_nvfp4_known_vector_geometry_swizzle_tail_and_round_trip():
    shape = (128, 64)
    geometry = block_scale_geometry("nvfp4", shape)
    assert (
        geometry.code_plane_bytes,
        geometry.scale_plane_offset,
        geometry.scale_plane_bytes,
        geometry.weight_divisor_offset,
        geometry.payload_bytes,
    ) == (4096, 4096, 512, 4608, 4612)

    packed = (
        torch.arange(geometry.code_plane_bytes, dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(128, 32)
    )
    packed[0, 0] = 0x10
    scales = (
        torch.arange(128 * 4, dtype=torch.int64)
        .remainder(0x7F)
        .to(torch.uint8)
        .reshape(128, 4)
    )
    divisor = struct.pack("<f", 2.5)
    payload = encode_nvfp4(packed, scales, divisor, shape)

    assert len(payload) == 4612
    assert payload[0] == 0x10  # low nibble is K=0; high nibble is K=1.
    for row, lane in ((0, 0), (31, 3), (32, 0), (127, 3)):
        offset = geometry.scale_plane_offset + (row % 32) * 16 + (row // 32) * 4 + lane
        assert payload[offset] == int(scales[row, lane])
    assert payload[geometry.weight_divisor_offset :] == divisor

    decoded_packed, decoded_scales, decoded_divisor = decode_nvfp4_words(payload, shape)
    assert torch.equal(decoded_packed, packed)
    assert torch.equal(decoded_scales, scales)
    assert bytes(decoded_divisor.reshape(1).view(torch.uint8).numpy()) == divisor


def test_ternary_pq2_known_vector_block_bytes_geometry_and_round_trip():
    shape = (2, 256)
    geometry = ternary_pq2_geometry("ternary_pq2_0", shape)
    assert (
        geometry.block_bytes,
        geometry.block_array_bytes,
        geometry.rotation_offset,
        geometry.rotation_bytes,
        geometry.payload_bytes,
    ) == (34, 136, 256, 16 + 256 * 4, 1296)

    # Worked example: scale 0.5 stores as binary16 0x3800 -> little-endian bytes 00 38.
    # Codes 00, 01, 10, 01 (low bits first) pack into one 0x64 byte; the remaining code
    # bytes are 0x55 (four 01 codes: all-zero values).
    codes = torch.full((2, 64), 0x55, dtype=torch.uint8)
    codes[0, 0] = 0x64
    scales = torch.tensor([0.5, 1.0, 1.0, 1.0], dtype=torch.float16).reshape(2, 2)
    signs = torch.ones(256, dtype=torch.float32)
    signs[128:] = -1.0
    payload = encode_ternary_pq2(codes, scales, signs, shape)

    assert len(payload) == 1296
    assert payload[0:2] == b"\x00\x38"
    assert payload[2] == 0x64
    assert payload[3:34] == b"\x55" * 31
    assert payload[34:36] == b"\x00\x3c"  # second block scale 1.0 -> 0x3C00.
    assert payload[36:68] == b"\x55" * 32
    assert payload[136:256] == b"\x00" * 120  # 256-byte alignment gap.

    header = struct.unpack_from("<IBBBBII", payload, 256)
    assert header == (1024, 0, 0, 0, 0, 0, 0)
    assert payload[272:276] == struct.pack("<f", 1.0)
    assert payload[784:788] == struct.pack("<f", -1.0)  # first -1.0 sign at lane 128.

    decoded_codes, decoded_scales, decoded_signs, rotation = decode_ternary_pq2_words(
        payload, shape
    )
    assert torch.equal(decoded_codes, codes)
    assert torch.equal(decoded_scales, scales)
    assert torch.equal(decoded_signs, signs)
    assert (rotation.block_size, rotation.transform, rotation.gdn_v_grouped, rotation.inverse) == (
        1024,
        0,
        False,
        False,
    )

    with pytest.raises(ValueError):
        encode_ternary_pq2(codes, torch.full((2, 2), float("nan"), dtype=torch.float16), signs, shape)
    with pytest.raises(ValueError):
        encode_ternary_pq2(codes, scales, torch.full((256,), 2.0, dtype=torch.float32), shape)
