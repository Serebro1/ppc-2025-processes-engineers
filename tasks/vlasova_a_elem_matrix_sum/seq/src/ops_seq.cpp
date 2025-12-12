#include "vlasova_a_elem_matrix_sum/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "vlasova_a_elem_matrix_sum/common/include/common.hpp"

namespace vlasova_a_elem_matrix_sum {

VlasovaAElemMatrixSumSEQ::VlasovaAElemMatrixSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool VlasovaAElemMatrixSumSEQ::ValidationImpl() {
  int rows = std::get<1>(GetInput());
  int cols = std::get<2>(GetInput());
  const auto &matrix_data = std::get<0>(GetInput());

  return matrix_data.size() == static_cast<size_t>(rows) * static_cast<size_t>(cols) && rows > 0 && cols > 0;
}

bool VlasovaAElemMatrixSumSEQ::PreProcessingImpl() {
  int rows = std::get<1>(GetInput());
  GetOutput().resize(rows, 0);
  return true;
}

bool VlasovaAElemMatrixSumSEQ::RunImpl() {
  int rows = std::get<1>(GetInput());
  int cols = std::get<2>(GetInput());
  const auto &matrix_data = std::get<0>(GetInput());

  for (int i = 0; i < rows; ++i) {
    int row_sum = 0;
    for (int j = 0; j < cols; ++j) {
      row_sum += matrix_data[(i * cols) + j];
    }
    GetOutput()[i] = row_sum;
  }
  return true;
}

bool VlasovaAElemMatrixSumSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace vlasova_a_elem_matrix_sum
