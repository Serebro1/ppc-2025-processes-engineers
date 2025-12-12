#pragma once

#include <string>
#include <tuple>
#include <vector>  // Добавлено для std::vector

#include "task/include/task.hpp"

namespace ivanova_p_max_matrix {

using InType = std::vector<std::vector<int>>;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace ivanova_p_max_matrix
