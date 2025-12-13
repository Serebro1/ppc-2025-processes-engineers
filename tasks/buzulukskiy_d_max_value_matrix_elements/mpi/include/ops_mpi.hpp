#pragma once

#include "buzulukskiy_d_max_value_matrix_elements/common/include/common.hpp"
#include "task/include/task.hpp"

namespace buzulukskiy_d_max_value_matrix_elements {

class BuzulukskiyDMaxValueMatrixElementsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BuzulukskiyDMaxValueMatrixElementsMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace buzulukskiy_d_max_value_matrix_elements
