#pragma once

#include "golovanov_d_matrix_max_elem//common/include/common.hpp"
#include "task/include/task.hpp"

namespace golovanov_d_matrix_max_elem {

class GolovanovDMatrixMaxElemMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit GolovanovDMatrixMaxElemMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace golovanov_d_matrix_max_elem
