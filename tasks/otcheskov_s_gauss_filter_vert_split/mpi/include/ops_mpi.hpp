#pragma once

#include <cstdint>
#include <vector>

#include "otcheskov_s_gauss_filter_vert_split/common/include/common.hpp"
#include "task/include/task.hpp"

namespace otcheskov_s_gauss_filter_vert_split {

class OtcheskovSGaussFilterVertSplitMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit OtcheskovSGaussFilterVertSplitMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  size_t GetLocalIndex(size_t row, size_t local_col, size_t channel, size_t width);

  void DistributeData();
  void ExchangeBoundaryColumns();

  void ApplyGaussianFilter();

  void CollectResults();

  bool is_valid_{};
  size_t proc_rank_{};
  size_t proc_num_{};
  size_t active_procs_{};

  size_t channels_{};
  size_t local_width_{};
  size_t start_col_{};
  size_t local_data_count_{};

  std::vector<uint8_t> local_data_;
  std::vector<uint8_t> local_output_;
  std::vector<uint8_t> extended_data_;
};

}  // namespace otcheskov_s_gauss_filter_vert_split
