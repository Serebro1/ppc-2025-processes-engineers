#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace shemetov_d_find_error_vec {

using InType = std::vector<double>;
using OutType = int;
using TestType = std::tuple<std::vector<double>, int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace shemetov_d_find_error_vec
