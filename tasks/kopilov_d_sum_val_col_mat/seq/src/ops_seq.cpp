#include "kopilov_d_sum_val_col_mat/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "kopilov_d_sum_val_col_mat/common/include/common.hpp"

namespace kopilov_d_sum_val_col_mat {

KopilovDSumValColMatSEQ::KopilovDSumValColMatSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool KopilovDSumValColMatSEQ::ValidationImpl() {
  const int rows = GetInput().rows;
  const int cols = GetInput().cols;
  const std::size_t expected_size = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  if (rows <= 0 || cols <= 0) {
    return false;
  }
  if (GetInput().data.size() != expected_size) {
    return false;
  }
  return true;
}

bool KopilovDSumValColMatSEQ::PreProcessingImpl() {
  const int cols = GetInput().cols;
  GetOutput().col_sum.assign(static_cast<std::size_t>(cols), 0.0);
  return true;
}

bool KopilovDSumValColMatSEQ::RunImpl() {
  const InType input = GetInput();
  const int rows = input.rows;
  const int cols = input.cols;
  const std::vector<double> &matrix = input.data;

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      GetOutput().col_sum[static_cast<std::size_t>(col)] +=
          matrix[(static_cast<std::size_t>(row) * static_cast<std::size_t>(cols)) + static_cast<std::size_t>(col)];
    }
  }

  return true;
}

bool KopilovDSumValColMatSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kopilov_d_sum_val_col_mat
