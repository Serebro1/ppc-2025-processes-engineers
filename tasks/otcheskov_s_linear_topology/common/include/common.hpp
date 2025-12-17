#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace otcheskov_s_linear_topology {

struct MessageHeader {
  int delivered{};
  int src{};
  int dest{};
  int data_size{};
};

using MessageData = std::vector<int>;
using Message = std::pair<MessageHeader, MessageData>;

using InType = Message;
using OutType = Message;
using TestType = std::tuple<Message, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace otcheskov_s_linear_topology
