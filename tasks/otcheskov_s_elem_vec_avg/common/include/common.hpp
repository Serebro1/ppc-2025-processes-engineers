#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace otcheskov_s_elem_vec_avg {

using InType = std::vector<int>;
using OutType = int;
using TestType = std::tuple<std::string, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace otcheskov_s_elem_vec_av
