// ninfer::ops - candidate_slice_softmax wrapper: public contract validation and launcher dispatch.
#include "ninfer/ops/candidate_slice_softmax.h"

#include "ops/launcher/candidate_slice_softmax.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops {
namespace {

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t columns,
                    const char* label) {
    if (tensor.ne[0] != rows || tensor.ne[1] != columns || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("candidate_slice_softmax: ") + label +
                                    " must have shape [N,C]");
    }
}

void require_accessible(const Tensor& tensor, std::size_t alignment, const char* label) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("candidate_slice_softmax: ") + label +
                                    " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string("candidate_slice_softmax: ") + label +
                                    " data must be non-null");
    }
    if ((reinterpret_cast<std::uintptr_t>(tensor.data) & (alignment - 1)) != 0) {
        throw std::invalid_argument(std::string("candidate_slice_softmax: ") + label +
                                    " data is not naturally aligned");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    if (lhs_begin <= rhs_begin) { return rhs_begin - lhs_begin < lhs.bytes(); }
    return lhs_begin - rhs_begin < rhs.bytes();
}

void check_cuda(cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("candidate_slice_softmax: ") +
                                 "host validation copy failed: " + cudaGetErrorString(err));
    }
}

} // namespace

void candidate_slice_softmax(const Tensor& logits, const Tensor& ids, const Tensor& groups,
                             float temperature, Tensor& out, cudaStream_t stream) {
    if (logits.dtype != DType::FP32) {
        throw std::invalid_argument("candidate_slice_softmax: logits must be FP32");
    }
    if (ids.dtype != DType::I32) {
        throw std::invalid_argument("candidate_slice_softmax: ids must be I32");
    }
    if (groups.dtype != DType::I32) {
        throw std::invalid_argument("candidate_slice_softmax: groups must be I32");
    }
    if (out.dtype != DType::FP32) {
        throw std::invalid_argument("candidate_slice_softmax: out must be FP32");
    }

    if (logits.ne[0] <= 0 || logits.ne[1] <= 0 || logits.ne[2] != 1 || logits.ne[3] != 1) {
        throw std::invalid_argument(
            "candidate_slice_softmax: logits must be rank-2 with positive dimensions");
    }
    const std::int32_t rows    = logits.ne[0];
    const std::int32_t columns = logits.ne[1];
    if (columns > 256) {
        throw std::invalid_argument("candidate_slice_softmax: logits must have 1<=C<=256 columns");
    }
    require_matrix(ids, rows, columns, "ids");
    require_matrix(groups, rows, columns, "groups");
    require_matrix(out, rows, columns, "out");

    if (!std::isfinite(temperature)) {
        throw std::invalid_argument("candidate_slice_softmax: temperature must be finite");
    }

    (void)logits.bytes();
    (void)ids.bytes();
    (void)groups.bytes();
    (void)out.bytes();
    require_accessible(logits, alignof(float), "logits");
    require_accessible(ids, alignof(std::int32_t), "ids");
    require_accessible(groups, alignof(std::int32_t), "groups");
    require_accessible(out, alignof(float), "out");
    if (overlaps(out, logits) || overlaps(out, ids) || overlaps(out, groups)) {
        throw std::invalid_argument("candidate_slice_softmax: out must not overlap any input");
    }

    // Non-finite + domain host validation: copy to the host on the caller's stream and check.
    const std::int64_t count = static_cast<std::int64_t>(rows) * columns;
    std::vector<float> host_logits(static_cast<std::size_t>(count));
    std::vector<std::int32_t> host_ids(static_cast<std::size_t>(count));
    std::vector<std::int32_t> host_groups(static_cast<std::size_t>(count));
    check_cuda(cudaMemcpyAsync(host_logits.data(), logits.data, host_logits.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    check_cuda(cudaMemcpyAsync(host_ids.data(), ids.data, host_ids.size() * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream));
    check_cuda(cudaMemcpyAsync(
        host_groups.data(), groups.data, host_groups.size() * sizeof(std::int32_t),
        cudaMemcpyDeviceToHost, stream));
    check_cuda(cudaStreamSynchronize(stream));
    for (std::int64_t i = 0; i < count; ++i) {
        const auto index = static_cast<std::size_t>(i);
        if (!std::isfinite(host_logits[index])) {
            throw std::invalid_argument("candidate_slice_softmax: logits must be finite");
        }
        const std::int32_t group = host_groups[index];
        if (group < -1 || group >= columns) {
            throw std::invalid_argument("candidate_slice_softmax: groups must be -1 or in [0,C)");
        }
        if (host_ids[index] < 0) {
            throw std::invalid_argument("candidate_slice_softmax: ids must be non-negative");
        }
    }

    detail::candidate_slice_softmax_launch(logits, ids, groups, temperature, out, stream);
}

} // namespace ninfer::ops
