#pragma once

#include "iskhakov_d_trapezoidal_integration/common/include/common.hpp"
#include "task/include/task.hpp"

namespace iskhakov_d_trapezoidal_integration {

class IskhakovDTrapezoidalIntegrationSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit IskhakovDTrapezoidalIntegrationSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace iskhakov_d_trapezoidal_integration
