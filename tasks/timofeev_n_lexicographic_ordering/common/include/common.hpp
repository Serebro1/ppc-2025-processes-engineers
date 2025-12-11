#pragma once

#include <string>
#include <tuple>
#include <utility>

#include "task/include/task.hpp"

namespace timofeev_n_lexicographic_ordering {

using InType = std::pair<std::string, std::string>;
/// @brief 1 - ordered, 0 - not ordered
using OutType = std::pair<int, int>;
using TestType = std::tuple<std::pair<int, int>, std::pair<std::string, std::string>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace timofeev_n_lexicographic_ordering
