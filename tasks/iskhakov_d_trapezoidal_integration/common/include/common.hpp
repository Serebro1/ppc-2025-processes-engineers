#pragma once

#include <functional>
#include <tuple>

#include "task/include/task.hpp"

namespace iskhakov_d_trapezoidal_integration {

using InType = std::tuple<double, double, std::function<double(double)>, int>;
using OutType = double;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace iskhakov_d_trapezoidal_integration
