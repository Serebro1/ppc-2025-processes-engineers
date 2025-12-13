#pragma once

#include "task/include/task.hpp"
#include "vlasova_a_elem_matrix_sum/common/include/common.hpp"

namespace vlasova_a_elem_matrix_sum {

class VlasovaAElemMatrixSumMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit VlasovaAElemMatrixSumMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace vlasova_a_elem_matrix_sum
