#pragma once

#include "shemetov_d_find_error_vec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shemetov_d_find_error_vec {

class ShemetovDFindErrorVecMPI : public BaseTask {
 public:
  explicit ShemetovDFindErrorVecMPI(const InType &input);

  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static int DetectDrop(double left, double right) noexcept;
};

}  // namespace shemetov_d_find_error_vec
