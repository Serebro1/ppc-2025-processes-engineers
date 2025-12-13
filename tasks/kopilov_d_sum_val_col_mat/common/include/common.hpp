#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kopilov_d_sum_val_col_mat {

struct Input {
  std::vector<double> data;
  int rows = 0;
  int cols = 0;
};

struct Output {
  std::vector<double> col_sum;
};

using InType = Input;
using OutType = Output;

using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kopilov_d_sum_val_col_mat
