#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace otcheskov_s_linear_topology {

struct Message {
  int src;
  int dest;
  std::vector<int> data;
};

struct Response {
  bool delivered;
  int hops;
  std::vector<int> path;
  std::vector<int> received_data{};
  int orig_src;
  int final_dest;
};

using InType = Message;
using OutType = Response;
using TestType = std::tuple<Message, Response>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace otcheskov_s_linear_topology
