#include "otcheskov_s_elem_vec_avg/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "otcheskov_s_elem_vec_avg/common/include/common.hpp"
#include "util/include/util.hpp"

namespace otcheskov_s_elem_vec_avg {

OtcheskovSElemVecAvgSEQ::OtcheskovSElemVecAvgSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = NAN;
}

bool OtcheskovSElemVecAvgSEQ::ValidationImpl() {
  return (!GetInput().empty() && std::isnan(GetOutput()));
}

bool OtcheskovSElemVecAvgSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool OtcheskovSElemVecAvgSEQ::RunImpl() {
  int sum = std::accumulate(GetInput().begin(), GetInput().end(), 0);
  GetOutput() = sum / static_cast<double>(GetInput().size());
  return true;
}

bool OtcheskovSElemVecAvgSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace otcheskov_s_elem_vec_avg
