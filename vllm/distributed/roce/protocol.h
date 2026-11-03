// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace vllm::roce {

// One registered slot, with independent send and receive lifetimes. GPU-side
// consumers must release a receive before begin_receive grants its next credit.
class ExchangeState {
 public:
  ExchangeState(uint32_t capacity, uint32_t row_bytes, uint32_t qps)
      : capacity_(capacity), row_bytes_(row_bytes), qps_(qps) {
    check(row_bytes && capacity % row_bytes == 0 && qps && qps <= 8,
          "Invalid transfer shape");
  }

  void begin_receive() {
    check(!receiving, "Receive slot is still active");
    advance(receive_epoch);
    received_mask_ = 0;
    receiving = true;
  }

  void begin_send(uint32_t bytes) {
    check(!sending, "Send slot is still active");
    validate_bytes(bytes);
    advance(send_epoch);
    send_bytes = bytes;
    sent_mask_ = empty_mask(bytes);
    sending = true;
    posted = false;
  }

  void accept_credit(uint64_t epoch) {
    check(peer_credit_epoch_ != std::numeric_limits<uint64_t>::max() &&
              epoch == peer_credit_epoch_ + 1 && epoch > 0 &&
              epoch - 1 <= send_epoch,
          "Duplicate, stale or premature receive credit");
    peer_credit_epoch_ = epoch;
  }

  void accept_descriptor(uint64_t epoch, uint32_t bytes) {
    validate_bytes(bytes);
    check(peer_descriptor_epoch_ != std::numeric_limits<uint64_t>::max() &&
              epoch == peer_descriptor_epoch_ + 1 && epoch > 0 &&
              epoch - 1 <= receive_epoch,
          "Duplicate, stale or premature payload descriptor");
    peer_bytes_[epoch % 2] = bytes;
    peer_descriptor_epoch_ = epoch;
    if (receiving && epoch == receive_epoch)
      check(!(received_mask_ & empty_mask(bytes)), "Unexpected empty stripe");
  }

  bool can_post() const {
    return sending && !posted && peer_credit_epoch_ >= send_epoch;
  }

  void mark_posted() {
    check(can_post(), "Send has no receive credit");
    posted = true;
  }

  void complete_send_stripe(uint32_t q) {
    check(sending && posted && q < qps_ && !(sent_mask_ & (1u << q)),
          "Unexpected or duplicate send completion");
    sent_mask_ |= 1u << q;
  }

  void complete_receive_stripe(uint32_t q) {
    check(receiving && q < qps_ && !(received_mask_ & (1u << q)),
          "Unexpected or duplicate receive completion");
    if (peer_descriptor_epoch_ >= receive_epoch)
      check(!(empty_mask(receive_bytes()) & (1u << q)),
            "Unexpected empty receive stripe");
    received_mask_ |= 1u << q;
  }

  bool send_complete() const {
    return sending && posted && sent_mask_ == all_mask();
  }

  bool receive_complete() const {
    return receiving && peer_descriptor_epoch_ >= receive_epoch &&
           (received_mask_ | empty_mask(receive_bytes())) == all_mask();
  }

  uint32_t receive_bytes() const {
    check(receiving && peer_descriptor_epoch_ >= receive_epoch,
          "Receive length is not published");
    return peer_bytes_[receive_epoch % 2];
  }

  void finish_send() {
    check(send_complete(), "Send is incomplete");
    sending = false;
  }

  void finish_receive() {
    check(receive_complete(), "Receive is incomplete");
    receiving = false;
  }

  uint64_t send_epoch = 0, receive_epoch = 0;
  uint32_t send_bytes = 0;
  bool sending = false, receiving = false, posted = false;

 private:
  static void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
  }

  static void advance(uint64_t& epoch) {
    check(epoch != std::numeric_limits<uint64_t>::max(), "RDMA epoch overflow");
    ++epoch;
  }

  void validate_bytes(uint32_t bytes) const {
    check(bytes <= capacity_ && bytes % row_bytes_ == 0,
          "Payload exceeds the registered shape");
  }

  uint32_t empty_mask(uint32_t bytes) const {
    uint32_t mask = 0;
    for (uint32_t q = 0; q < qps_; ++q)
      if (uint64_t(bytes) * q / qps_ == uint64_t(bytes) * (q + 1) / qps_)
        mask |= 1u << q;
    return mask;
  }

  uint32_t all_mask() const { return (1u << qps_) - 1; }
  uint32_t capacity_, row_bytes_, qps_, sent_mask_ = 0, received_mask_ = 0;
  uint64_t peer_credit_epoch_ = 0, peer_descriptor_epoch_ = 0;
  std::array<uint32_t, 2> peer_bytes_{};
};
}  // namespace vllm::roce
