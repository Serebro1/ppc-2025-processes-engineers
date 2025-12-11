#include "zorin_d_avg_vec/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "zorin_d_avg_vec/common/include/common.hpp"

namespace zorin_d_avg_vec {

ZorinDAvgVecSEQ::ZorinDAvgVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ZorinDAvgVecSEQ::ValidationImpl() {
  return true;
}

bool ZorinDAvgVecSEQ::PreProcessingImpl() {
  return true;
}

bool ZorinDAvgVecSEQ::RunImpl() {
  const auto &vec = GetInput();
  if (vec.empty()) {
    GetOutput() = 0.0;
    return true;
  }

  const double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
  GetOutput() = sum / static_cast<double>(vec.size());

  return true;
}

bool ZorinDAvgVecSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zorin_d_avg_vec
