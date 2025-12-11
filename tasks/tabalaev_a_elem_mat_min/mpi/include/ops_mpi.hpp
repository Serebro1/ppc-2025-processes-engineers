#pragma once

#include "tabalaev_a_elem_mat_min/common/include/common.hpp"
#include "task/include/task.hpp"

namespace tabalaev_a_elem_mat_min {

class TabalaevAElemMatMinMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TabalaevAElemMatMinMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace tabalaev_a_elem_mat_min
