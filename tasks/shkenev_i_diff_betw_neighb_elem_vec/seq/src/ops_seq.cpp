#include "shkenev_i_diff_betw_neighb_elem_vec/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "shkenev_i_diff_betw_neighb_elem_vec/common/include/common.hpp"

namespace shkenev_i_diff_betw_neighb_elem_vec {

ShkenevIDiffBetwNeighbElemVecSEQ::ShkenevIDiffBetwNeighbElemVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ShkenevIDiffBetwNeighbElemVecSEQ::ValidationImpl() {
  return true;
}

bool ShkenevIDiffBetwNeighbElemVecSEQ::PreProcessingImpl() {
  return true;
}

bool ShkenevIDiffBetwNeighbElemVecSEQ::RunImpl() {
  const std::vector<int> &vec = GetInput();
  std::size_t n = vec.size();
  if (n < 2) {
    GetOutput() = 0;
    return true;
  }

  int max_diff = 0;
  for (std::size_t i = 0; i < n - 1; i++) {
    int diff = std::abs(vec[i + 1] - vec[i]);
    max_diff = std::max(diff, max_diff);
  }

  GetOutput() = max_diff;
  return true;
}

bool ShkenevIDiffBetwNeighbElemVecSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace shkenev_i_diff_betw_neighb_elem_vec
