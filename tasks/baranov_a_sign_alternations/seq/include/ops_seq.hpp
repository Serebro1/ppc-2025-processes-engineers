#pragma once

#include "baranov_a_sign_alternations/common/include/common.hpp"
#include "task/include/task.hpp"

namespace baranov_a_sign_alternations {

class BaranovASignAlternationsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit BaranovASignAlternationsSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace baranov_a_sign_alternations
