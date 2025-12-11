#include "klimenko_v_max_matrix_elems_val/seq/include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <vector>

#include "klimenko_v_max_matrix_elems_val/common/include/common.hpp"

namespace klimenko_v_max_matrix_elems_val {

KlimenkoVMaxMatrixElemsValSEQ::KlimenkoVMaxMatrixElemsValSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = 0;
}

bool KlimenkoVMaxMatrixElemsValSEQ::ValidationImpl() {
  if (GetInput().empty() || GetInput()[0].empty()) {
    GetOutput() = 0;
  }
  return true;
}

bool KlimenkoVMaxMatrixElemsValSEQ::PreProcessingImpl() {
  return GetOutput() == 0;
}

bool KlimenkoVMaxMatrixElemsValSEQ::RunImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    GetOutput() = 0;
    return true;
  }

  int max_element = matrix[0][0];
  for (const auto &row : matrix) {
    for (int element : row) {
      max_element = std::max(element, max_element);
    }
  }

  GetOutput() = max_element;
  return true;
}

bool KlimenkoVMaxMatrixElemsValSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace klimenko_v_max_matrix_elems_val
