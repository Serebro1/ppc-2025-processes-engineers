#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace tsyplakov_k_vec_neighbours {

using InType = std::vector<int>;       // вектор
using OutType = std::tuple<int, int>;  // мин. разность
using TestType = std::tuple<std::vector<int>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace tsyplakov_k_vec_neighbours
