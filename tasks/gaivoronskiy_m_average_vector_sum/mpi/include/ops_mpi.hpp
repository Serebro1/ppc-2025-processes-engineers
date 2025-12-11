#pragma once

#include <cstddef>
#include <vector>

#include "gaivoronskiy_m_average_vector_sum/common/include/common.hpp"
#include "task/include/task.hpp"

namespace gaivoronskiy_m_average_vector_sum {

class GaivoronskiyMAverageVecSumMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit GaivoronskiyMAverageVecSumMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  InType distributed_values_;
  std::vector<double> local_buffer_;
  std::size_t total_size_ = 0;
  double local_sum_ = 0.0;
  double global_sum_ = 0.0;
  int world_rank_ = 0;
  int world_size_ = 1;
};

}  // namespace gaivoronskiy_m_average_vector_sum
