#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace otcheskov_s_linear_topology {

struct Message {
  int src;
  int dest;
  std::vector<int> data;
  bool delivered;
};

using InType = Message;
using OutType = Message;
using TestType = std::tuple<Message, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace otcheskov_s_linear_topology
