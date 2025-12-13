#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace golovanov_d_matrix_max_elem {

using InType = std::tuple<int, int, std::vector<double>>;
using OutType = double;
using TestType = std::tuple<int, int, double>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace golovanov_d_matrix_max_elem
