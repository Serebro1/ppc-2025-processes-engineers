#include "ivanova_p_max_matrix/seq/include/ops_seq.hpp"

#include <algorithm>  // для std::max
#include <cstddef>    // для size_t
#include <limits>
#include <vector>

#include "ivanova_p_max_matrix/common/include/common.hpp"

namespace ivanova_p_max_matrix {

IvanovaPMaxMatrixSEQ::IvanovaPMaxMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  // Безопасная инициализация
  GetInput().clear();
  if (!in.empty()) {
    GetInput() = in;
  }

  GetOutput() = std::numeric_limits<int>::min();
}

bool IvanovaPMaxMatrixSEQ::ValidationImpl() {
  // Базовая проверка на пустоту
  if (GetInput().empty()) {
    return false;
  }

  const size_t cols = GetInput()[0].size();
  return std::all_of(GetInput().begin(), GetInput().end(),
                     [cols](const std::vector<int> &row) { return row.size() == cols; });
}

bool IvanovaPMaxMatrixSEQ::PreProcessingImpl() {
  // Инициализируем выход минимальным значением
  GetOutput() = std::numeric_limits<int>::min();
  return true;
}

bool IvanovaPMaxMatrixSEQ::RunImpl() {
  int max_val = std::numeric_limits<int>::min();

  for (const auto &row : GetInput()) {
    for (int val : row) {
      max_val = std::max(val, max_val);
    }
  }

  GetOutput() = max_val;
  return true;
}

bool IvanovaPMaxMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace ivanova_p_max_matrix
