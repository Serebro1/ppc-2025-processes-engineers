#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace klimenko_v_max_matrix_elems_val {

using InType = std::vector<std::vector<int>>;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace klimenko_v_max_matrix_elems_val
