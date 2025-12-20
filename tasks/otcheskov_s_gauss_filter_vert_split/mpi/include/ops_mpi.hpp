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

  size_t GetLocalIndex(int row, size_t local_col, int channel, size_t width);
  void BroadcastImgMetadata();

  void DistributeData();
  void ExchangeBoundaryColumns();

  void ApplyGaussianFilter();
  int MirrorRow(int row, int height);

  void CollectResults();

  int proc_rank_{};
  int proc_num_{};
  int active_procs_{};

  int channels_{};
  size_t local_width_{};
  size_t start_col_{};
  size_t local_data_count_{};

  std::vector<uint8_t> local_data_;
  std::vector<uint8_t> local_output_;
  std::vector<uint8_t> extended_data_;
};

}  // namespace otcheskov_s_gauss_filter_vert_split
