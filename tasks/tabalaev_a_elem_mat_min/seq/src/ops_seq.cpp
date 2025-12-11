#include "tabalaev_a_elem_mat_min/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "tabalaev_a_elem_mat_min/common/include/common.hpp"

namespace tabalaev_a_elem_mat_min {

TabalaevAElemMatMinSEQ::TabalaevAElemMatMinSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool TabalaevAElemMatMinSEQ::ValidationImpl() {
  auto &rows = std::get<0>(GetInput());
  auto &columns = std::get<1>(GetInput());
  auto &matrix = std::get<2>(GetInput());

  return (rows != 0 && columns != 0) && (rows * columns == matrix.size()) && (GetOutput() == 0);
}

bool TabalaevAElemMatMinSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool TabalaevAElemMatMinSEQ::RunImpl() {
  auto &matrix = std::get<2>(GetInput());

  if (matrix.empty()) {
    return false;
  }

  int minik = matrix[0];
  for (size_t i = 1; i < matrix.size(); i++) {
    minik = std::min(minik, matrix[i]);
  }

  GetOutput() = minik;
  return true;
}

bool TabalaevAElemMatMinSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace tabalaev_a_elem_mat_min
