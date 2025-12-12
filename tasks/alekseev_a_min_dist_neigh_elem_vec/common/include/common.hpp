#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace alekseev_a_min_dist_neigh_elem_vec {

using InType = std::vector<int>;
using OutType = std::tuple<int, int>;
using TestType = std::tuple<std::vector<int>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace alekseev_a_min_dist_neigh_elem_vec
