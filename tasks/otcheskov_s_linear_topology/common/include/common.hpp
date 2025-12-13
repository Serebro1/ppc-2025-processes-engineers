#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace otcheskov_s_linear_topology {

struct MessageHeader {
  bool delivered{};
  int src{};
  int dest{};
  int data_size{};
};

struct Message {
  int src{};
  int dest{};
  std::vector<int> data{};
  bool delivered{};
};

using InType = Message;
using OutType = Message;
using TestType = std::tuple<Message, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace otcheskov_s_linear_topology
