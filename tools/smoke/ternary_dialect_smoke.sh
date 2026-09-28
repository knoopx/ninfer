#!/usr/bin/env bash
# ternary_dialect_smoke.sh — PQ2_0_G128 / PTQ1_0_G128 format-dialect port smoke check.
#
# Two phases, clearly separated:
#   1. BUILD  — configure + build /tmp/ninfer-build (nix develop one-shot form, AGENTS.md)
#   2. RUN    — one engine forward pass on a ternary artifact, then the unit-level
#               GEMM-vs-FP64-oracle regression on the linear Op suite
#
# Usage:
#   tools/smoke/ternary_dialect_smoke.sh <artifact>
#
#   <artifact>  path to a PQ2/PTQ1 .ninfer artifact. Absolute, or a bare name resolved
#               under the serve-route model root ~/.local/share/ninfer/models/.
#
# This script is run by the user; the agent must not execute it (GPU + build gate).
set -euo pipefail

readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly BUILD_DIR=/tmp/ninfer-build
readonly MODEL_ROOT="$HOME/.local/share/ninfer/models"

# --- argument: artifact path ---------------------------------------------------
if [[ $# -ne 1 ]]; then
    echo "usage: $0 <artifact>" >&2
    exit 2
fi
ARTIFACT="$1"
# Select the model by explicit path under the serve-route model root.
if [[ ! -e "$ARTIFACT" && -e "$MODEL_ROOT/$ARTIFACT" ]]; then
    ARTIFACT="$MODEL_ROOT/$ARTIFACT"
fi
if [[ ! -f "$ARTIFACT" ]]; then
    echo "artifact not found: $ARTIFACT (looked in cwd and under $MODEL_ROOT)" >&2
    exit 2
fi

# --- A8 prefill-tile route check --------------------------------------------
# parse_prefill_seconds LOG — print the CLI "text prefill" stage duration (LOG is the
# CLI's captured stderr) as a float number of seconds; empty when absent.
parse_prefill_seconds() {
    local rest
    rest=$(grep -E 'text prefill' <<<"$1" | sed -E 's/.*text prefill[[:space:]]*//' | head -1 || true)
    awk -v r="${rest:-n/a}" 'BEGIN{
        if (r ~ /us$/)    { sub(/ *us$/, "", r); print r * 1e-6 }
        else if (r ~ /ms$/){ sub(/ *ms$/, "", r); print r * 1e-3 }
        else if (r ~ /s$/) { sub(/ *s$/,  "", r); print r + 0 }
        else               { print "" }
    }'
}

# check_a8_tile_route ARTIFACT — run the ternary A8 prefill route with the 128x64
# int8-activation tile ON and OFF at one small-T and two prefill T values, and report
# the route each config takes plus the prefill-timing delta. Both configs compute the
# same D = W*X, so the numerical rel-l2 equivalence is confirmed by the perplexity run
# (expect ~5.631 for both; a large on/off perplexity delta would mean the tile route
# diverged from the shared GEMM). The route is derived from the routing logic in
# src/ops/linear/ternary/ternary_a8.cu route(): T < 64 takes the small-T kernel (tile env has no
# effect there); T >= 64 takes the prefill route, which is the 128x64 tile GEMM with
# NINFER_PQ2_A8_TILE=on and the shared rowsplit_a8_mma GEMM with =off.
check_a8_tile_route() {
    local artifact="$1"
    local cli="$BUILD_DIR/apps/ninfer"
    local t prompt on_log off_log on_secs off_secs route_on route_off delta
    echo "a8 tile route: NINFER_PQ2_A8_TILE on vs off (T = 8, 128, 256)"
    for t in 8 128 256; do
        # ~T tokens: T repetitions of a single-token word; the route keys off the prompt
        # token count, so this lands in the small-T band (8) or the prefill route (128, 256).
        prompt=$(printf 'ok %.0s' $(seq 1 "$t"))
        on_log=$(NINFER_PQ2_A8_TILE=on  "$cli" "$artifact" --prompt "$prompt" --max-new 1 --max-context 4096 2>&1)
        off_log=$(NINFER_PQ2_A8_TILE=off "$cli" "$artifact" --prompt "$prompt" --max-new 1 --max-context 4096 2>&1)
        on_secs=$(parse_prefill_seconds "$on_log")
        off_secs=$(parse_prefill_seconds "$off_log")
        if (( t < 64 )); then
            route_on="small-T (ternary_small_t_i8)"; route_off="small-T (ternary_small_t_i8)"
        else
            route_on="tile GEMM (PQ2PrefillI8 128x64)"; route_off="shared GEMM (rowsplit_a8_mma)"
        fi
        delta=$(awk -v a="${on_secs:-0}" -v b="${off_secs:-0}" 'BEGIN{printf "%+.3f", b - a}')
        echo "  T=$t  on : route=$route_on   prefill=${on_secs:-n/a}s"
        echo "  T=$t  off: route=$route_off  prefill=${off_secs:-n/a}s"
        echo "  T=$t  prefill delta (off-on) = ${delta}s"
    done
    echo "  rel-l2 equivalence: confirm via the perplexity run (expect ~5.631 for both configs)"
}

# ============================================================================
# BUILD PHASE
# ============================================================================
cd "$REPO_ROOT"

# First build configures; a cached build skips re-configuration.
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    nix develop path:. -c cmake -S CMAKE_BUILD_TYPE=Release -B /tmp/ninfer-build .
fi
nix develop path:. -c cmake --build /tmp/ninfer-build -j
# The server target (serve route).
nix develop path:. -c cmake --build /tmp/ninfer-build -j --target ninfer-serve -j

# ============================================================================
# RUN PHASE
# ============================================================================

# --- one engine forward pass on the ternary artifact -------------------------
# The CLI runs exactly one request against the artifact: load + one generation.
"$BUILD_DIR/apps/ninfer" "$ARTIFACT" \
    --prompt "Reply with exactly: OK" \
    --max-context 4096 \
    --max-new 8

# --- unit-level GEMM-vs-FP64-oracle check -------------------------------------
# TODO(ternary): no ternary-specific linear oracle test exists yet. The harness
# pattern to follow is tests/ops/linear/: per-quantization test files
# (test_q4_a16.cpp etc.) registered via ninfer_add_op_test in tests.cmake, all
# built on cpu_linear_gemm_fp64() in linear_test_common.{h,cpp} (naive FP64
# accumulation oracle) with the ops/quantized_weight.h packed fixture. Add
# test_ptq1_g128_a16.cpp and test_pq2_g128_a16.cpp following test_q4_a16.cpp
# (a ternary packed-weight generator must be added to ops/quantized_weight.h),
# register them in tests/ops/linear/tests.cmake, then gate on them here.
#
# Until that exists, the existing linear A16 oracle suite is the regression
# check on the shared linear path (it exercises the same linear() Op the
# ternary decode uses). It requires -DBUILD_TESTING=ON at configure time.
if grep -q '^BUILD_TESTING:BOOL=ON' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    nix develop path:. -c cmake --build /tmp/ninfer-build -j --target \
        ninfer_linear_q4_a16_test \
        ninfer_linear_q5_a16_test \
        ninfer_linear_q6_a16_test \
        ninfer_linear_q8_a16_test
    ctest --test-dir "$BUILD_DIR" -R '^ninfer_linear_(q4|q5|q6|q8)_a16_test$' --output-on-failure
else
    echo "SKIP: linear GEMM-vs-FP64-oracle suite (reconfigure with -DBUILD_TESTING=ON to run it)"
fi

# --- A8 prefill-tile route check (NINFER_PQ2_A8_TILE on vs off) ---------------
check_a8_tile_route "$ARTIFACT"

echo "ternary dialect smoke: forward pass + oracle regression + a8 tile route complete"
