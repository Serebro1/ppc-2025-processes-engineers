#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace vlasova_a_elem_matrix_sum {

using InType = std::tuple<std::vector<int>, int, int>;
using OutType = std::vector<int>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace vlasova_a_elem_matrix_sum
