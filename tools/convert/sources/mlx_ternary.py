"""MLX Prism Hadamard ternary PQ2_0 source: model.safetensors plus hadamard.json.

Maps NInfer logical parameters onto the MLX package's ternary matrices (U32
codes, F16 scales/biases, F32 signs) and full-precision F32 tensors. Ternary
matrices yield encoded rows with zero code transcoding and zero dequantization:
the little-endian U32 words are the block code bytes verbatim, and the
rotation auxiliary carries the 16-byte header plus the matrix's F32 signs.
"""

from __future__ import annotations

import json
from math import prod
from pathlib import Path
import re
from typing import Mapping

import torch

from tools.artifact.codecs.ternary_pq2 import (
    TERNARY_PQ2_TRANSFORM_SYLVESTER,
    _rotation_header,
)
from tools.artifact.formats import (
    valid_ternary_pq2_scale_word,
    valid_ternary_pq2_sign_word,
)
from tools.artifact.layouts import TERNARY_PQ2_ROTATION_BLOCK_SIZE

from .logical import EncodedRows, LogicalSource, select_rows
from .safetensors import SafetensorsSource

_PREFIX = "language_model."
_IGNORED_PREFIX = "vision_tower."

#: (MLX tensor suffix, NInfer logical parameter names, transpose axes, ternary).
#: The layer index placeholder substitutes the package's contiguous layer block.
_MAP_PATTERNS: tuple[tuple[str, tuple[str, ...], tuple[int, ...] | None, bool], ...] = (
    ("lm_head.weight", ("text/output_head",), None, True),
    ("model.embed_tokens.weight", ("text/token_embedding",), None, True),
    ("model.norm.weight", ("text/final_norm",), None, False),
    (
        "model.layers.{i}.input_layernorm.weight",
        ("text/layers/{i}/input_norm",),
        None,
        False,
    ),
    (
        "model.layers.{i}.post_attention_layernorm.weight",
        ("text/layers/{i}/post_attention_norm",),
        None,
        False,
    ),
    (
        "model.layers.{i}.self_attn.q_proj.weight",
        (
            "text/layers/{i}/attention/query",
            "text/layers/{i}/attention/gate",
        ),
        None,
        True,
    ),
    (
        "model.layers.{i}.self_attn.k_proj.weight",
        ("text/layers/{i}/attention/key",),
        None,
        True,
    ),
    (
        "model.layers.{i}.self_attn.v_proj.weight",
        ("text/layers/{i}/attention/value",),
        None,
        True,
    ),
    (
        "model.layers.{i}.self_attn.q_norm.weight",
        ("text/layers/{i}/attention/query_norm",),
        None,
        False,
    ),
    (
        "model.layers.{i}.self_attn.k_norm.weight",
        ("text/layers/{i}/attention/key_norm",),
        None,
        False,
    ),
    (
        "model.layers.{i}.self_attn.o_proj.weight",
        ("text/layers/{i}/attention/output",),
        None,
        True,
    ),
    (
        "model.layers.{i}.linear_attn.A_log",
        ("text/layers/{i}/gdn/a_log",),
        None,
        False,
    ),
    (
        "model.layers.{i}.linear_attn.dt_bias",
        ("text/layers/{i}/gdn/dt_bias",),
        None,
        False,
    ),
    (
        "model.layers.{i}.linear_attn.conv1d.weight",
        ("text/layers/{i}/gdn/convolution",),
        (2, 0, 1),
        False,
    ),
    (
        "model.layers.{i}.linear_attn.in_proj_a.weight",
        ("text/layers/{i}/gdn/a_projection",),
        None,
        False,
    ),
    (
        "model.layers.{i}.linear_attn.in_proj_b.weight",
        ("text/layers/{i}/gdn/b_projection",),
        None,
        False,
    ),
    (
        "model.layers.{i}.linear_attn.in_proj_qkv.weight",
        (
            "text/layers/{i}/gdn/query",
            "text/layers/{i}/gdn/key",
            "text/layers/{i}/gdn/value",
        ),
        None,
        True,
    ),
    (
        "model.layers.{i}.linear_attn.in_proj_z.weight",
        ("text/layers/{i}/gdn/z",),
        None,
        True,
    ),
    (
        "model.layers.{i}.linear_attn.norm.weight",
        ("text/layers/{i}/gdn/norm",),
        None,
        False,
    ),
    (
        "model.layers.{i}.linear_attn.out_proj.weight",
        ("text/layers/{i}/gdn/output",),
        None,
        True,
    ),
    (
        "model.layers.{i}.mlp.gate_proj.weight",
        ("text/layers/{i}/mlp/gate",),
        None,
        True,
    ),
    (
        "model.layers.{i}.mlp.up_proj.weight",
        ("text/layers/{i}/mlp/up",),
        None,
        True,
    ),
    (
        "model.layers.{i}.mlp.down_proj.weight",
        ("text/layers/{i}/mlp/down",),
        None,
        True,
    ),
)


def _decode_ternary_values(
    words: torch.Tensor, scales: torch.Tensor, width: int
) -> torch.Tensor:
    """Exact FP64 values of ternary rows: trit x binary16 group scale.

    ``words`` is ``[rows, width/16]`` little-endian U32 code words (read as int32)
    and ``scales`` is ``[rows, width/128]`` binary16. The 8 words of a 128-group
    are the 32 GGUF code bytes verbatim, so the decode is a pure bit unpack:
    weight lane ``l`` owns code byte ``l//4`` (word ``(l//4)//4``, byte
    ``(l//4)%4``), bits ``2*(l%4)..2*(l%4)+1`` low bits first, codebook
    ``00->-1, 01->0, 10->+1, 11->+2``.
    """
    rows = words.shape[0]
    groups = width // 128
    lane = torch.arange(128, dtype=torch.int32)
    raw = words.view(torch.uint8).reshape(rows, groups, 8, 4).to(torch.int32)
    code = (raw[:, :, lane // 16, (lane // 4) % 4] >> ((lane % 4) * 2)) & 3
    trits = code - 1  # {0,1,2,3} -> {-1,0,1,2}
    values = trits.to(torch.float64) * scales.reshape(rows, groups).to(torch.float64).unsqueeze(-1)
    return values.reshape(rows, width)


def _sylvester_fwt1024(values: torch.Tensor) -> torch.Tensor:
    """Sylvester Walsh-Hadamard over the last 1024 lanes in the caller's dtype.

    The pinned fork butterfly (docs/maintainer/ternary-pq2-0.md section 3, kernel
    reference): within a pair (j, j^h) with j < j^h the low element takes
    x + y and the high element takes x - y.
    """
    if values.shape[-1] != 1024:
        raise ValueError("sylvester FWT requires 1024 lanes")
    x = values
    h = 1
    while h < 1024:
        depth = 1024 // (2 * h)
        x = x.view(*x.shape[:-1], depth, 2, h)
        low, high = x[..., 0, :], x[..., 1, :]
        x = torch.stack((low + high, low - high), dim=-2).reshape(values.shape)
        h *= 2
    return x


def unfold_ternary_rows(
    words: torch.Tensor, scales: torch.Tensor, signs: torch.Tensor, width: int
) -> torch.Tensor:
    """Plain-basis (un-rotated) FP64 values of folded ternary rows.

    The runtime kernel rotates activation rows per 1024 block as
    ``x' = (1/32) * FWT(x * s)`` (sign flip and 1/sqrt(1024) fused into the A-tile
    load), i.e. it computes ``x . ((1/32) H S)^T``. The plain head is the folded
    head right-multiplied by the same matrix, so each row becomes
    ``w' = (1/32) * FWT(w) * s``: the Sylvester FWT first, then the elementwise
    sign flip, per 1024 block of the K axis.
    """
    if width % 1024:
        raise ValueError(f"un-fold requires a 1024-multiple width, got {width}")
    blocks = width // 1024
    folded = _decode_ternary_values(words, scales, width)
    transformed = _sylvester_fwt1024(folded.reshape(-1, blocks, 1024))
    factor = (signs.to(torch.float64).reshape(blocks, 1024) / 32.0).expand_as(transformed)
    return (transformed * factor).reshape(words.shape[0], width)


class _MlxTernaryMatrix:
    """One folded/inverse ternary matrix with its validated auxiliaries."""

    def __init__(self, source: "MlxTernarySource", base: str, inverse: bool) -> None:
        self.base = base
        store = source._store
        weight = store.describe(base + ".weight")
        if weight.dtype != "U32" or len(weight.shape) != 2:
            raise ValueError(f"{base}.weight: expected a 2D U32 ternary matrix")
        self.rows, self.words_per_row = weight.shape
        self.width = self.words_per_row * 16
        if self.rows <= 0 or self.width % 128:
            raise ValueError(
                f"{base}: ternary width {self.width} is not a positive "
                "multiple of the 128-weight group"
            )
        groups = self.width // 128
        self.groups = groups
        for suffix in (".scales", ".biases"):
            info = store.describe(base + suffix)
            if info.dtype != "F16" or tuple(info.shape) != (self.rows, groups):
                raise ValueError(f"{base}{suffix}: expected F16 [{self.rows},{groups}]")
        signs_info = store.describe(base + ".signs")
        if signs_info.dtype != "F32" or tuple(signs_info.shape) != (self.width,):
            raise ValueError(f"{base}.signs: expected F32 [{self.width}]")
        scales = store.read_flat(base + ".scales")
        biases = store.read_flat(base + ".biases")
        if not torch.equal(biases, -scales):
            raise ValueError(f"{base}: biases are not exactly -scales; drop is unsafe")
        scale_words = scales.view(torch.int16).to(torch.int32) & 0xFFFF
        bad_scale = [
            word
            for word in scale_words.flatten().tolist()
            if not valid_ternary_pq2_scale_word(int(word))
        ]
        if bad_scale:
            raise ValueError(
                f"{base}: ternary scales must be finite binary16 words"
            )
        signs = store.read_flat(base + ".signs")
        sign_words = signs.view(torch.int32).to(torch.uint32).flatten().tolist()
        bad_sign = [
            word for word in sign_words if not valid_ternary_pq2_sign_word(int(word))
        ]
        if bad_sign:
            raise ValueError(
                f"{base}: rotation signs must be exactly +/-1.0 "
                f"(saw {hex(bad_sign[0])})"
            )
        self.signs = signs
        self.weight_divisor = (
            _rotation_header(
                TERNARY_PQ2_ROTATION_BLOCK_SIZE,
                TERNARY_PQ2_TRANSFORM_SYLVESTER,
                source._gdn_v_grouped,
                inverse,
            )
            + signs.numpy().tobytes()
        )

    def read_encoded(self, store: SafetensorsSource, begin: int, end: int) -> EncodedRows:
        if not 0 <= begin <= end <= self.rows:
            raise ValueError(
                f"{self.base}: encoded row range [{begin},{end}) exceeds {self.rows}"
            )
        rows = end - begin
        codes = store.read_flat(
            f"{self.base}.weight", begin * self.words_per_row, end * self.words_per_row
        )
        codes = codes.view(torch.uint8).reshape(rows, self.width // 4)
        scales = store.read_flat(
            f"{self.base}.scales", begin * self.groups, end * self.groups
        ).reshape(rows, self.groups)
        return EncodedRows(
            "ternary_pq2_0", codes, scales, self.weight_divisor
        )


class MlxTernarySource:
    """Context-managed MLX package source with row-chunked bounded reads."""

    def __init__(self, root: str | Path) -> None:
        self.path = Path(root)
        self.root = self.path
        self._store = SafetensorsSource(self.root / "model.safetensors")
        manifest = json.loads((self.root / "hadamard.json").read_text())
        self._manifest = self._validate_manifest(manifest)
        self._gdn_v_grouped = bool(manifest["prism.hadamard.gdn_v_grouped"])
        header_names = set(self._store.weight_map)
        (
            self._mlx_to_ninfer,
            self._matrix_for,
            self._direct_for,
            self._transposes,
            self._ternary_bases,
        ) = self._build_name_map(header_names)
        self._check_coverage(header_names)
        self._matrices: dict[str, _MlxTernaryMatrix] = {}
        self._direct_values: dict[str, torch.Tensor] = {}

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "MlxTernarySource":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @staticmethod
    def _validate_manifest(manifest: Mapping) -> Mapping:
        def expect(key: str, value) -> None:
            if manifest.get(key) != value:
                raise ValueError(
                    f"hadamard manifest {key!r}: expected {value!r}, "
                    f"got {manifest.get(key)!r}"
                )

        expect("prism.hadamard.version", 1)
        block_size = manifest.get("prism.hadamard.block_size")
        if type(block_size) is not int or block_size != TERNARY_PQ2_ROTATION_BLOCK_SIZE:
            raise ValueError(
                "hadamard manifest block size must be "
                f"{TERNARY_PQ2_ROTATION_BLOCK_SIZE}"
            )
        expect("prism.hadamard.transform", "normalized-sylvester-walsh-hadamard")
        expect("prism.hadamard.axis", "input-last-dimension")
        expect("prism.hadamard.sign_mode", "explicit")
        if type(manifest.get("prism.hadamard.gdn_v_grouped")) is not bool:
            raise ValueError("hadamard manifest gdn_v_grouped must be boolean")
        weight_names = manifest.get("prism.hadamard.weight_names")
        inverse_names = manifest.get("prism.hadamard.inverse_weight_names")
        if (
            not isinstance(weight_names, list)
            or not weight_names
            or not all(type(name) is str for name in weight_names)
        ):
            raise ValueError("hadamard manifest weight_names must be a nonempty list")
        if not isinstance(inverse_names, list) or not all(
            type(name) is str for name in inverse_names
        ):
            raise ValueError(
                "hadamard manifest inverse_weight_names must be a list of names"
            )
        if set(weight_names) & set(inverse_names):
            raise ValueError(
                "hadamard manifest lists a matrix as both folded and inverse"
            )
        widths = manifest.get("prism.hadamard.sign_widths")
        values = manifest.get("prism.hadamard.sign_values")
        if not isinstance(widths, list) or not widths:
            raise ValueError("hadamard manifest sign_widths must be a nonempty list")
        for width in widths:
            if type(width) is not int or width <= 0 or width % block_size:
                raise ValueError(
                    f"hadamard sign width {width!r} must be a positive "
                    f"multiple of the block size {block_size}"
                )
        if not isinstance(values, list) or len(values) != sum(widths):
            raise ValueError(
                "hadamard sign_values length must equal the sum of sign_widths"
            )
        for value in values:
            if value not in (1.0, -1.0):
                raise ValueError("hadamard sign values must be exactly +/-1")
        return manifest

    def _build_name_map(self, header_names: set[str]):
        layers = sorted(
            {
                int(match.group(1))
                for name in header_names
                if (match := re.match(r"^" + _PREFIX + r"model\.layers\.(\d+)\.", name))
            }
        )
        if layers != list(range(len(layers))):
            raise ValueError(f"MLX layer indices are not contiguous from zero: {layers}")
        mlx_to_ninfer: dict[str, tuple[str, ...]] = {}
        matrix_for: dict[str, str] = {}
        direct_for: dict[str, str] = {}
        transposes: dict[str, tuple[int, ...]] = {}
        ternary_bases: set[str] = set()

        def register(tensor: str, targets: tuple[str, ...]) -> None:
            previous = mlx_to_ninfer.get(tensor)
            if previous is not None and previous != targets:
                raise ValueError(f"MLX tensor {tensor!r} maps to conflicting parameters")
            mlx_to_ninfer[tensor] = targets

        for mlx_suffix, ninfer_suffixes, transpose, ternary in _MAP_PATTERNS:
            base_suffix = (
                mlx_suffix[: -len(".weight")] if ternary else mlx_suffix
            )
            indices = layers if "{i}" in base_suffix else (None,)
            for index in indices:
                key = "" if index is None else str(index)
                base = _PREFIX + base_suffix.replace("{i}", key)
                tensor = base + ".weight" if ternary else base
                if tensor not in header_names:
                    continue
                targets = tuple(
                    name.replace("{i}", key) for name in ninfer_suffixes
                )
                for target in targets:
                    if ternary:
                        if target in matrix_for:
                            raise ValueError(
                                f"{target}: multiple MLX matrices claim this parameter"
                            )
                        matrix_for[target] = base
                    else:
                        if target in direct_for:
                            raise ValueError(
                                f"{target}: multiple MLX tensors claim this parameter"
                            )
                        direct_for[target] = base
                if ternary:
                    ternary_bases.add(base)
                    register(base + ".weight", targets)
                    for companion in (".scales", ".biases", ".signs"):
                        register(base + companion, targets)
                else:
                    register(base, targets)
                if transpose is not None:
                    transposes[base] = transpose
        return (
            mlx_to_ninfer,
            matrix_for,
            direct_for,
            transposes,
            ternary_bases,
        )

    def _check_coverage(self, header_names: set[str]) -> None:
        unknown = sorted(
            name
            for name in header_names
            if not name.startswith(_IGNORED_PREFIX)
            and name not in self._mlx_to_ninfer
        )
        if unknown:
            raise ValueError(
                f"MLX package has {len(unknown)} unmapped language tensors, "
                f"e.g. {unknown[:4]}"
            )
        expected_ternary = (
            set(self._manifest["prism.hadamard.weight_names"])
            | set(self._manifest["prism.hadamard.inverse_weight_names"])
        )
        mapped_ternary = {base + ".weight" for base in self._ternary_bases}
        if missing := expected_ternary - mapped_ternary:
            raise ValueError(f"hadamard manifest matrices missing from the map: {sorted(missing)[:4]}")
        if extra := mapped_ternary - expected_ternary:
            raise ValueError(f"mapped matrices missing from the manifest: {sorted(extra)[:4]}")
        for name in expected_ternary:
            if name not in header_names:
                raise ValueError(f"hadamard manifest matrix {name!r} is absent from the package")

    def _matrix(self, base: str, inverse: bool) -> _MlxTernaryMatrix:
        matrix = self._matrices.get(base)
        if matrix is None:
            matrix = _MlxTernaryMatrix(self, base, inverse)
            self._matrices[base] = matrix
        return matrix

    def ternary(
        self,
        param_name: str,
        shape: tuple[int, ...],
        rows: tuple[tuple[int, int], ...] | None = None,
    ) -> LogicalSource:
        base = self._matrix_for.get(param_name)
        if base is None:
            raise ValueError(f"{param_name}: no MLX ternary matrix is mapped")
        inverse = base + ".weight" in set(
            self._manifest["prism.hadamard.inverse_weight_names"]
        )
        matrix = self._matrix(base, inverse)

        def read_values(begin: int, end: int) -> torch.Tensor:
            raise ValueError(
                f"mlx-ternary:{base}: values are unavailable; use encoded rows"
            )

        def read_encoded(begin: int, end: int) -> EncodedRows:
            return matrix.read_encoded(self._store, begin, end)

        if rows is not None:
            source = LogicalSource(
                (matrix.rows, matrix.width),
                f"mlx-ternary:{base}",
                read_values,
                read_encoded,
            )
            source = select_rows(source, rows)
        else:
            source = LogicalSource(
                (matrix.rows, matrix.width),
                f"mlx-ternary:{base}",
                read_values,
                read_encoded,
            )
        if tuple(source.shape) != tuple(shape):
            raise ValueError(
                f"{param_name}: MLX source shape {source.shape} differs from {shape}"
            )
        return source

    def rebase_head_q8(self, param_name: str, shape: tuple[int, ...]) -> LogicalSource:
        """Plain-basis FP64 values of a folded ternary head for grouped Q8 storage.

        The consumer (linear_topk) admits only full-row Q8_G32_FP16 heads, while the
        MLX package stores the head folded in the Hadamard-rotated basis. This source
        decodes the exact trit x binary16 rows and undoes the kernel's K-axis rotation
        (see unfold_ternary_rows) so the head can be stored un-rotated as the
        registered Q8_G32_FP16 format via grouped_absmax.
        """
        base = self._matrix_for.get(param_name)
        if base is None:
            raise ValueError(f"{param_name}: no MLX ternary matrix is mapped")
        if base + ".weight" in set(self._manifest["prism.hadamard.inverse_weight_names"]):
            raise ValueError(f"{param_name}: inverse-transform matrices cannot be rebased")
        matrix = self._matrix(base, inverse=False)
        if len(shape) != 2 or tuple(shape) != (matrix.rows, matrix.width):
            raise ValueError(
                f"{param_name}: rebased head shape {shape} differs from "
                f"{(matrix.rows, matrix.width)}"
            )
        width = matrix.width
        words_per_row = matrix.words_per_row
        store = self._store
        signs = matrix.signs

        def read_values(begin: int, end: int) -> torch.Tensor:
            if not 0 <= begin <= end <= matrix.rows * width:
                raise ValueError(
                    f"{param_name}: rebased range [{begin},{end}) exceeds "
                    f"{matrix.rows * width}"
                )
            first_row, last_row = begin // width, (end + width - 1) // width
            words = store.read_flat(
                f"{base}.weight", first_row * words_per_row, last_row * words_per_row
            ).reshape(last_row - first_row, words_per_row)
            scales = store.read_flat(
                f"{base}.scales", first_row * matrix.groups, last_row * matrix.groups
            ).reshape(last_row - first_row, matrix.groups)
            values = unfold_ternary_rows(words, scales, signs, width).reshape(-1)
            return values[begin - first_row * width : end - first_row * width]

        return LogicalSource(
            (matrix.rows, width),
            f"mlx-ternary-rebased:{base}",
            read_values,
        )

    def direct(self, param_name: str, shape: tuple[int, ...]) -> LogicalSource:
        tensor = self._direct_for.get(param_name)
        if tensor is None:
            raise ValueError(f"{param_name}: no MLX full-precision tensor is mapped")
        transpose = self._transposes.get(tensor)
        info = self._store.describe(tensor)

        if transpose is not None:
            label = f"mlx-direct:{tensor}[t{transpose}]"

            def read(begin: int, end: int) -> torch.Tensor:
                data = self._direct_values.get(tensor)
                if data is None:
                    data = (
                        self._store.read_flat(tensor)
                        .reshape(info.shape)
                        .permute(transpose)
                        .contiguous()
                        .reshape(-1)
                    )
                    self._direct_values[tensor] = data
                return data[begin:end]
        else:
            label = f"mlx-direct:{tensor}"

            def read(begin: int, end: int) -> torch.Tensor:
                return self._store.read_flat(tensor, begin, end)

        if prod(info.shape) != prod(shape):
            raise ValueError(
                f"{param_name}: MLX tensor {tensor} has {prod(info.shape)} "
                f"values, expected {prod(shape)}"
            )
        return LogicalSource(tuple(shape), label, read)
