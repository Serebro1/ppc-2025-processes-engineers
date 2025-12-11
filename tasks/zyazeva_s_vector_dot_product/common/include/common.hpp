#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace zyazeva_s_vector_dot_product {

using InType = std::vector<std::vector<int>>;
using OutType = int64_t;
using TestType = std::tuple<int64_t, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace zyazeva_s_vector_dot_product
