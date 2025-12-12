#pragma once

#include "popova_e_integr_monte_carlo/common/include/common.hpp"
#include "task/include/task.hpp"

namespace popova_e_integr_monte_carlo {

class PopovaEIntegrMonteCarloMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PopovaEIntegrMonteCarloMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int point_count_{};
  double a_{};
  double b_{};
  FuncType func_id_{};
};

}  // namespace popova_e_integr_monte_carlo
