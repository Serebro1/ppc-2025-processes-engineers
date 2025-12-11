#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace gaivoronskiy_m_average_vector_sum {

using InType = std::vector<double>;
using OutType = double;
using TestType = std::tuple<std::string, double>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace gaivoronskiy_m_average_vector_sum
