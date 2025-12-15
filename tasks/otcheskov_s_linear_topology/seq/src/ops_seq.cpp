#include "otcheskov_s_linear_topology/seq/include/ops_seq.hpp"

#include <cmath>
#include <vector>

#include "otcheskov_s_linear_topology/common/include/common.hpp"

namespace otcheskov_s_linear_topology {

OtcheskovSLinearTopologySEQ::OtcheskovSLinearTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool OtcheskovSLinearTopologySEQ::ValidationImpl() {
  return GetInput().src >= 0 && GetInput().dest >= 0 && !GetInput().delivered && !GetInput().data.empty();
}

bool OtcheskovSLinearTopologySEQ::PreProcessingImpl() {
  return true;
}

bool OtcheskovSLinearTopologySEQ::RunImpl() {
  GetOutput() = GetInput();
  GetOutput().delivered = true;
  return true;
}

bool OtcheskovSLinearTopologySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace otcheskov_s_linear_topology
