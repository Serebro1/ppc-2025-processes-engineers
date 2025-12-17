#include "otcheskov_s_linear_topology/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "otcheskov_s_linear_topology/common/include/common.hpp"

namespace otcheskov_s_linear_topology {

OtcheskovSLinearTopologySEQ::OtcheskovSLinearTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool OtcheskovSLinearTopologySEQ::ValidationImpl() {
  const auto &header = GetInput().first;
  const auto &data = GetInput().second;
  return header.src >= 0 && header.dest >= 0 && header.delivered == 0 && !data.empty() &&
         static_cast<size_t>(header.data_size) == data.size();
}

bool OtcheskovSLinearTopologySEQ::PreProcessingImpl() {
  return true;
}

bool OtcheskovSLinearTopologySEQ::RunImpl() {
  GetOutput() = GetInput();
  GetOutput().first.delivered = 1;
  return true;
}

bool OtcheskovSLinearTopologySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace otcheskov_s_linear_topology
