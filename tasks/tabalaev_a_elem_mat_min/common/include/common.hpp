#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace tabalaev_a_elem_mat_min {

using InType = std::tuple<size_t, size_t, std::vector<int>>;
using OutType = int;
using TestType = std::tuple<int, int, int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace tabalaev_a_elem_mat_min
