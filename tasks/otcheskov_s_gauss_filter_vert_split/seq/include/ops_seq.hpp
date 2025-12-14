#pragma once

#include "otcheskov_s_gauss_filter_vert_split/common/include/common.hpp"
#include "task/include/task.hpp"

namespace otcheskov_s_gauss_filter_vert_split {

class OtcheskovSGaussFilterVertSplitSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit OtcheskovSGaussFilterVertSplitSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int GetIndex(int row, int col, int channel);
};

}  // namespace otcheskov_s_gauss_filter_vert_split
