#include "golovanov_d_matrix_max_elem//seq/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

#include "golovanov_d_matrix_max_elem//common/include/common.hpp"

namespace golovanov_d_matrix_max_elem {

GolovanovDMatrixMaxElemSEQ::GolovanovDMatrixMaxElemSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GolovanovDMatrixMaxElemSEQ::ValidationImpl() {
  int columns = std::get<0>(GetInput());
  int strokes = std::get<1>(GetInput());
  return (columns > 0) && (strokes > 0) && (static_cast<int>(std::get<2>(GetInput()).size()) == (strokes * columns)) &&
         (GetOutput() == 0);
}

bool GolovanovDMatrixMaxElemSEQ::PreProcessingImpl() {
  return true;
}

bool GolovanovDMatrixMaxElemSEQ::RunImpl() {
  std::vector<double> elems = std::get<2>(GetInput());
  double max = *std::ranges::max_element(elems);
  GetOutput() = max;
  return true;
}

bool GolovanovDMatrixMaxElemSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace golovanov_d_matrix_max_elem
