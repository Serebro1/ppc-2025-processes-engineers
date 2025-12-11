#include "gaivoronskiy_m_average_vector_sum/seq/include/ops_seq.hpp"

#include <cmath>
#include <numeric>

#include "gaivoronskiy_m_average_vector_sum/common/include/common.hpp"

namespace gaivoronskiy_m_average_vector_sum {

GaivoronskiyMAverageVecSumSEQ::GaivoronskiyMAverageVecSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool GaivoronskiyMAverageVecSumSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool GaivoronskiyMAverageVecSumSEQ::PreProcessingImpl() {
  data_ = GetInput();
  partial_sum_ = 0.0;
  return !data_.empty();
}

bool GaivoronskiyMAverageVecSumSEQ::RunImpl() {
  partial_sum_ = std::accumulate(data_.begin(), data_.end(), 0.0);
  return true;
}

bool GaivoronskiyMAverageVecSumSEQ::PostProcessingImpl() {
  GetOutput() = partial_sum_ / static_cast<double>(data_.size());
  return std::isfinite(GetOutput());
}

}  // namespace gaivoronskiy_m_average_vector_sum
