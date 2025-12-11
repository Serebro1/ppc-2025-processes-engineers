#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace zorin_d_avg_vec {

using InType = std::vector<int>;
using OutType = double;
using TestType = std::tuple<InType, OutType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace zorin_d_avg_vec
