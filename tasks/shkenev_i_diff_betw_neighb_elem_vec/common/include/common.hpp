#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace shkenev_i_diff_betw_neighb_elem_vec {

using InType = std::vector<int>;
using OutType = int;
using TestType = std::tuple<std::vector<int>, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace shkenev_i_diff_betw_neighb_elem_vec
