#include "buzulukskiy_d_max_value_matrix_elements/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "buzulukskiy_d_max_value_matrix_elements/common/include/common.hpp"

namespace buzulukskiy_d_max_value_matrix_elements {

BuzulukskiyDMaxValueMatrixElementsSEQ::BuzulukskiyDMaxValueMatrixElementsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool BuzulukskiyDMaxValueMatrixElementsSEQ::ValidationImpl() {
  const Matrix &inputdata = GetInput();
  const int rows = inputdata.rows;
  const int columns = inputdata.columns;
  const std::vector<int> &matrix = inputdata.data;

  return !matrix.empty() && rows > 0 && columns > 0 &&
         matrix.size() == static_cast<size_t>(rows) * static_cast<size_t>(columns);
}

bool BuzulukskiyDMaxValueMatrixElementsSEQ::PreProcessingImpl() {
  return true;
}

bool BuzulukskiyDMaxValueMatrixElementsSEQ::RunImpl() {
  const Matrix &inputdata = GetInput();
  const std::vector<int> &matrix = inputdata.data;

  // Обязательная проверка на пустую матрицу
  if (matrix.empty()) {
    GetOutput() = 0;
    return true;
  }

  int max_value = matrix[0];
  for (size_t index = 1; index < matrix.size(); ++index) {
    max_value = std::max(max_value, matrix[index]);
  }

  GetOutput() = max_value;
  return true;
}

bool BuzulukskiyDMaxValueMatrixElementsSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace buzulukskiy_d_max_value_matrix_elements
