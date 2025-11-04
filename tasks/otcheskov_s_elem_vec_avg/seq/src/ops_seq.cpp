#include "otcheskov_s_elem_vec_avg/seq/include/ops_seq.hpp"

#include <cmath>

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
  return (!GetInput().empty() && std::isnan(GetOutput()));
}

bool OtcheskovSElemVecAvgSEQ::RunImpl() {
  if (GetInput().empty() || !std::isnan(GetOutput())) {
    return false;
  }
  int sum = std::accumulate(GetInput().begin(), GetInput().end(), 0);
  GetOutput() = sum / static_cast<double>(GetInput().size());
  return !std::isnan(GetOutput());
}

bool OtcheskovSElemVecAvgSEQ::PostProcessingImpl() {
  return (!GetInput().empty() && !std::isnan(GetOutput()));
}

}  // namespace otcheskov_s_elem_vec_avg
