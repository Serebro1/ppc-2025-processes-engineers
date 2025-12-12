#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace buzulukskiy_d_max_value_matrix_elements {

struct Matrix {
  std::vector<int> data;
  int rows;
  int columns;
};

using InType = Matrix;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace buzulukskiy_d_max_value_matrix_elements
