#pragma once

#include "shemetov_d_find_error_vec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shemetov_d_find_error_vec {

class ShemetovDFindErrorVecSEQ : public BaseTask {
 public:
  explicit ShemetovDFindErrorVecSEQ(const InType &input);

  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace shemetov_d_find_error_vec
