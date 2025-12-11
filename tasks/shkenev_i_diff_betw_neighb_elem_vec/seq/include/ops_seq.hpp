#pragma once

#include "shkenev_i_diff_betw_neighb_elem_vec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shkenev_i_diff_betw_neighb_elem_vec {

class ShkenevIDiffBetwNeighbElemVecSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ShkenevIDiffBetwNeighbElemVecSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace shkenev_i_diff_betw_neighb_elem_vec
