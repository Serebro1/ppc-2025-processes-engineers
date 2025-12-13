#pragma once

#include "ivanova_p_max_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ivanova_p_max_matrix {

class IvanovaPMaxMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit IvanovaPMaxMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace ivanova_p_max_matrix
