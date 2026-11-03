// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "vllm/distributed/roce/protocol.h"

#include <algorithm>
#include <cassert>
#include <random>
#include <string>

using vllm::roce::ExchangeState;

template <typename F>
void rejects(F fn) {
  bool rejected = false;
  try {
    fn();
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
}

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::string name = argv[1];
  if (name == "receive_before_local_producer") {
    ExchangeState source(64, 8, 4), destination(64, 8, 4);
    destination.begin_receive();
    source.accept_credit(1);
    source.begin_send(32);
    assert(source.can_post());
    source.mark_posted();
    destination.accept_descriptor(1, 32);
    for (uint32_t q = 0; q < 4; ++q) destination.complete_receive_stripe(q);
    assert(destination.receive_complete());
    assert(!destination.sending && destination.send_epoch == 0);
    assert(!source.send_complete());
    assert(destination.receive_bytes() == 32);
    destination.finish_receive();
    for (uint32_t q = 0; q < 4; ++q) source.complete_send_stripe(q);
    assert(source.send_complete() && !source.receiving);
    source.finish_send();
  } else if (name == "descriptor_after_data") {
    ExchangeState state(64, 8, 4);
    state.begin_receive();
    for (uint32_t q = 0; q < 4; ++q) state.complete_receive_stripe(q);
    assert(!state.receive_complete());
    rejects([&] { state.finish_receive(); });
    state.accept_descriptor(1, 64);
    assert(state.receive_complete());
    state.finish_receive();
  } else if (name == "zero_length_next_epoch") {
    ExchangeState state(64, 8, 4);
    state.begin_receive();
    state.accept_descriptor(1, 0);
    assert(state.receive_complete());
    state.accept_descriptor(2, 64);
    assert(state.receive_bytes() == 0);
    state.finish_receive();
    state.begin_receive();
    assert(!state.receive_complete() && state.receive_bytes() == 64);
    for (uint32_t q = 0; q < 4; ++q) state.complete_receive_stripe(q);
    assert(state.receive_complete());
  } else if (name == "invalid_reuse_and_messages") {
    ExchangeState state(64, 8, 4);
    rejects([&] { state.accept_credit(2); });
    state.accept_credit(1);
    rejects([&] { state.accept_credit(1); });
    rejects([&] { state.begin_send(65); });
    rejects([&] { state.begin_send(1); });
    state.begin_send(64);
    rejects([&] { state.begin_send(64); });
    rejects([&] { state.finish_send(); });
    state.mark_posted();
    state.complete_send_stripe(0);
    rejects([&] { state.complete_send_stripe(0); });
    state.begin_receive();
    rejects([&] { state.begin_receive(); });
    rejects([&] { state.accept_descriptor(3, 0); });
    rejects([&] { state.accept_descriptor(1, 72); });
    state.accept_descriptor(1, 0);
    rejects([&] { state.complete_receive_stripe(0); });
    rejects([&] { state.accept_descriptor(1, 0); });
  } else if (name == "changing_replays") {
    ExchangeState a(257, 1, 8), b(257, 1, 8);
    std::mt19937 random(42);
    for (uint64_t epoch = 1; epoch <= 10000; ++epoch) {
      a.begin_receive();
      b.begin_receive();
      a.accept_credit(epoch);
      b.accept_credit(epoch);
      const uint32_t a_bytes = random() % 258, b_bytes = random() % 258;
      a.begin_send(a_bytes);
      b.begin_send(b_bytes);
      a.mark_posted();
      b.mark_posted();
      b.accept_descriptor(epoch, a_bytes);
      std::array<uint32_t, 8> order{0, 1, 2, 3, 4, 5, 6, 7};
      std::shuffle(order.begin(), order.end(), random);
      for (auto q : order) {
        if (a_bytes * q / 8 != a_bytes * (q + 1) / 8) {
          a.complete_send_stripe(q);
          b.complete_receive_stripe(q);
        }
        if (b_bytes * q / 8 != b_bytes * (q + 1) / 8) {
          b.complete_send_stripe(q);
          a.complete_receive_stripe(q);
        }
      }
      assert(a.send_complete() && b.send_complete());
      assert(!a.receive_complete() && b.receive_complete());
      a.accept_descriptor(epoch, b_bytes);
      assert(a.receive_complete());
      assert(a.receive_bytes() == b_bytes && b.receive_bytes() == a_bytes);
      a.finish_send();
      b.finish_send();
      a.finish_receive();
      b.finish_receive();
    }
  } else {
    assert(false && "Unknown test case");
  }
}
