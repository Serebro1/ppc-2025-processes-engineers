#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace trofimov_n_max_val_matrix {

using InType = std::vector<std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace trofimov_n_max_val_matrix
