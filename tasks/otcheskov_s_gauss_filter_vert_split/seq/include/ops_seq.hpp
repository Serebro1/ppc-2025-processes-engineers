#pragma once

#include <cstddef>

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

  size_t GetIndex(size_t row, size_t col, size_t channel);
  size_t MirrorCoord(size_t current, int off, size_t size);

  bool is_valid_{};
};

}  // namespace otcheskov_s_gauss_filter_vert_split
