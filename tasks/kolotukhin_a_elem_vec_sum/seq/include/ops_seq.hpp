#pragma once

#include "kolotukhin_a_elem_vec_sum/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kolotukhin_a_elem_vec_sum {

class KolotukhinAElemVecSumSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KolotukhinAElemVecSumSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kolotukhin_a_elem_vec_sum
