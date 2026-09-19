#include "runtime/engine/admission_policy.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::runtime {
namespace {

[[nodiscard]] bool contains(std::span<const std::uint64_t> ids, std::uint64_t id) noexcept {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

[[nodiscard]] bool is_donor(const AdmissionProtection& protection, std::uint64_t id) noexcept {
    return contains(
        std::span<const std::uint64_t>(protection.donor_ids.data(), protection.donor_count), id);
}

void validate_active_partition(const AdmissionProtection& protection,
                               std::span<const ActiveAdmissionSnapshot> active) {
    if (active.size() > kMaximumConcurrency) {
        throw std::invalid_argument("protected admission exceeds startup concurrency");
    }
    for (std::size_t row = 0; row < active.size(); ++row) {
        const ActiveAdmissionSnapshot& request = active[row];
        if (request.request_id == 0 || request.request_id == protection.head_request_id) {
            throw std::invalid_argument("protected admission contains an invalid request");
        }
        for (std::size_t prior = 0; prior < row; ++prior) {
            if (active[prior].request_id == request.request_id) {
                throw std::invalid_argument("protected admission contains a duplicate request");
            }
        }
        if (is_donor(protection, request.request_id)) {
            if (request.backfill_epoch == protection.epoch_id &&
                (request.backfill_class == BackfillClass::Persistent ||
                 request.backfill_class == BackfillClass::Temporal)) {
                throw std::logic_error("frozen donor was relabelled as a protected-epoch borrower");
            }
            continue;
        }
        if (request.backfill_epoch != protection.epoch_id ||
            (request.backfill_class != BackfillClass::Persistent &&
             request.backfill_class != BackfillClass::Temporal)) {
            throw std::logic_error("new active request is outside protected-head ownership");
        }
    }
}

} // namespace

AdmissionProtection make_admission_protection(std::uint64_t epoch_id, std::uint64_t head_request_id,
                                              ProgramResourceRevision resource_revision,
                                              std::span<const ActiveAdmissionSnapshot> active) {
    if (epoch_id == 0 || head_request_id == 0 || resource_revision.value == 0 || active.empty() ||
        active.size() > kMaximumConcurrency) {
        throw std::invalid_argument("invalid protected-admission identity");
    }
    AdmissionProtection protection{
        .epoch_id          = epoch_id,
        .head_request_id   = head_request_id,
        .resource_revision = resource_revision,
    };
    std::array<std::size_t, kMaximumConcurrency> order{};
    for (std::size_t i = 0; i < active.size(); ++i) { order[i] = i; }
    std::sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(active.size()),
              [&](std::size_t lhs, std::size_t rhs) {
                  if (active[lhs].remaining_work_quanta != active[rhs].remaining_work_quanta) {
                      return active[lhs].remaining_work_quanta < active[rhs].remaining_work_quanta;
                  }
                  return active[lhs].request_id < active[rhs].request_id;
              });
    for (std::size_t i = 0; i < active.size(); ++i) {
        const ActiveAdmissionSnapshot& donor = active[order[i]];
        if (donor.request_id == 0 || donor.request_id == head_request_id ||
            donor.remaining_work_quanta == 0 ||
            contains(
                std::span<const std::uint64_t>(protection.donor_ids.data(), protection.donor_count),
                donor.request_id)) {
            throw std::invalid_argument("invalid protected-admission incumbent");
        }
        protection.donor_ids[protection.donor_count++] = donor.request_id;
        // Mint the credit from the shortest-remaining donor upward; the frozen donor set is
        // complete, so the final mint bounds the frontier at the last donor's remaining work.
        protection.temporal_credit = donor.remaining_work_quanta;
    }
    return protection;
}

void rebind_admission_protection(AdmissionProtection& protection,
                                 std::span<const ActiveAdmissionSnapshot> active,
                                 ProgramResourceRevision resource_revision) {
    if (protection.epoch_id == 0 || protection.head_request_id == 0 ||
        resource_revision.value == 0) {
        throw std::invalid_argument("invalid protected-admission rebind");
    }
    validate_active_partition(protection, active);
    protection.resource_revision = resource_revision;
}

bool protection_has_live_donor(const AdmissionProtection& protection,
                               std::span<const ActiveAdmissionSnapshot> active) noexcept {
    return std::any_of(active.begin(), active.end(), [&](const ActiveAdmissionSnapshot& request) {
        return is_donor(protection, request.request_id);
    });
}

bool persistent_backfill_is_authorized(const AdmissionProtection& protection,
                                       std::uint64_t candidate_request_id,
                                       std::span<const ActiveAdmissionSnapshot> active,
                                       ProgramResourceRevision program_proof_revision) noexcept {
    if (candidate_request_id == 0 || candidate_request_id == protection.head_request_id ||
        program_proof_revision.value == 0 ||
        program_proof_revision != protection.resource_revision ||
        !protection_has_live_donor(protection, active)) {
        return false;
    }
    for (const ActiveAdmissionSnapshot& request : active) {
        if (request.request_id == candidate_request_id) { return false; }
        if (!is_donor(protection, request.request_id) &&
            (request.backfill_epoch != protection.epoch_id ||
             request.backfill_class != BackfillClass::Persistent)) {
            return false;
        }
    }
    return true;
}

std::uint64_t
protection_frontier_distance(const AdmissionProtection& protection,
                             std::span<const ActiveAdmissionSnapshot> active) noexcept {
    std::uint64_t distance = 0;
    for (const ActiveAdmissionSnapshot& request : active) {
        if (is_donor(protection, request.request_id)) {
            distance = std::max(distance, request.remaining_work_quanta);
        }
    }
    return distance;
}

bool protected_head_safe_without_temporal(const AdmissionProtection& protection,
                                          std::span<const ActiveAdmissionSnapshot> active) noexcept {
    return !protection_has_live_donor(protection, active);
}

} // namespace ninfer::runtime
