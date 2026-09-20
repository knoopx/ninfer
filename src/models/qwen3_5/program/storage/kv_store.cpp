#include "models/qwen3_5/program/storage/kv_store.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

void KVAddressSpaceStore::prepare_branch_fork_batch(
    KVAddressSpaceHandle source_handle, const std::vector<KVAddressSpaceHandle>& destinations,
    std::uint32_t frontier, std::uint32_t entitlement, cudaStream_t stream) {
    const Address& source = require(source_handle);
    if (source.active || source.row || source.reservation.valid()) {
        throw std::logic_error("KV branch-fork source is still active");
    }
    if (destinations.empty()) { throw std::invalid_argument("KV branch fork has no destination"); }
    // Destination i binds execution row i + 1; row 0 stays with the source's caller.
    if (destinations.size() + 1U > static_cast<std::size_t>(tables_->row_count())) {
        throw std::logic_error("KV branch fork exceeds the execution-table row capacity");
    }
    for (const KVAddressSpaceHandle destination_handle : destinations) {
        if (destination_handle == source_handle) {
            throw std::logic_error("KV branch-fork destination aliases the source");
        }
        const Address& destination = require(destination_handle);
        if (destination.active || destination.row || destination.reservation.valid() ||
            destination.page_count != 0 || destination.committed_frontier != 0) {
            throw std::logic_error("KV branch-fork destination is not empty");
        }
    }
    if (frontier == 0 || frontier > source.committed_frontier) {
        throw std::logic_error("KV branch-fork frontier is unavailable");
    }
    const std::uint32_t required_pages = pages_for_tokens(frontier);
    if (entitlement < required_pages || entitlement > page_capacity_) {
        throw std::invalid_argument("KV branch-fork entitlement is invalid");
    }
    const std::uint32_t page_size    = static_cast<std::uint32_t>(kPagedKVPageSize);
    const std::uint32_t full_pages   = frontier / page_size;
    const std::uint32_t tail_columns = frontier % page_size;
    for (std::uint32_t page = 0; page < required_pages; ++page) {
        const LogicalKVPageHandle logical = membership(source, page);
        const std::uint32_t required      = page < full_pages ? page_size : tail_columns;
        if (!pages_->device_resident(logical) || pages_->committed_columns(logical) < required ||
            !pages_->can_pin_source(logical) ||
            (page < full_pages && !pages_->can_retain_reference(logical, false))) {
            throw std::logic_error("KV branch-fork source is not stable");
        }
    }
    for (std::uint32_t page = 0; page < required_pages; ++page) {
        pages_->pin_source(membership(source, page));
    }

    struct RowWork {
        DeviceKVPageReservation reservation;
        std::optional<KVExecutionRowLease> row;
        std::optional<LogicalKVPageHandle> tail_destination;
        bool committed = false;
    };
    std::vector<RowWork> work(destinations.size());
    try {
        for (std::size_t index = 0; index < destinations.size(); ++index) {
            Address& destination = require(destinations[index]);
            work[index].reservation = pages_->physical_pool().make_empty_reservation();
            try {
                pages_->physical_pool().resize_reservation(work[index].reservation,
                                                           entitlement - full_pages);
            } catch (const std::bad_alloc&) {
                const auto& pool = pages_->physical_pool();
                throw std::runtime_error(
                    "KV branch-fork page reservation failed (branch=" + std::to_string(index) +
                    " required=" + std::to_string(entitlement - full_pages) + " available=" +
                    std::to_string(pool.available_pages()) + " reserved=" +
                    std::to_string(pool.reserved_pages()) + " capacity=" +
                    std::to_string(pool.capacity_pages()) + ")");
            }
            work[index].row = tables_->acquire(static_cast<std::int32_t>(index) + 1);
            if (tail_columns != 0) {
                work[index].tail_destination =
                    pages_->materialize_transfer_destination(work[index].reservation,
                                                             tail_columns);
            }

            publish_scratch_.clear();
            for (std::uint32_t page = 0; page < full_pages; ++page) {
                publish_scratch_.push_back(pages_->physical(membership(source, page)));
            }
            if (work[index].tail_destination) {
                publish_scratch_.push_back(pages_->physical(*work[index].tail_destination));
            }
            tables_->publish(work[index].row->handle(), 0, publish_scratch_, stream);

            for (std::uint32_t page = 0; page < full_pages; ++page) {
                const LogicalKVPageHandle logical = membership(source, page);
                pages_->retain_reference(logical, false);
                pages_->protect_coverage(logical, page_size);
                membership(destination, page) = logical;
            }
            if (work[index].tail_destination) {
                pages_->publish_transfer_destination(*work[index].tail_destination, true);
                membership(destination, full_pages) = *work[index].tail_destination;
            }
            destination.page_count         = required_pages;
            destination.committed_frontier = frontier;
            destination.reservation        = std::move(work[index].reservation);
            destination.row.emplace(std::move(*work[index].row));
            destination.active = true;
            for (std::uint32_t page = 0; page < required_pages; ++page) {
                pages_->retain_active_reference(membership(destination, page));
            }
            work[index].committed = true;
        }
    } catch (...) {
        // Unwind before releasing the shared source pins: a committed row's release is only
        // valid once the forked reader references are the sole extra references.
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            const LogicalKVPageHandle logical = membership(source, page);
            if (pages_->source_pins(logical) != 0) { pages_->unpin_source(logical); }
        }
        for (std::size_t index = 0; index < destinations.size(); ++index) {
            if (work[index].committed) {
                (void)release_after_deactivate(destinations[index]);
            } else if (work[index].tail_destination) {
                pages_->abort_transfer_destination(*work[index].tail_destination,
                                                  work[index].reservation);
                (void)release(destinations[index]);
            }
        }
        throw;
    }
    for (std::uint32_t page = 0; page < required_pages; ++page) {
        pages_->unpin_source(membership(source, page));
    }
}

} // namespace ninfer::models::qwen3_5::detail
