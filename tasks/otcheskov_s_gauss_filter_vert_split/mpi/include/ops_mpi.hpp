#pragma once

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

  int GetGlobalIndex(int row, int col, int channel);
  int GetLocalIndex(int row, int local_col, int channel);
  void DistributeData();
  void ExchangeBoundaryColumns();
  void ApplyGaussianFilter();
  void CollectResults();

  int proc_rank_{};
  int proc_num_{};
  int local_width_{};
  int start_col_{};
  int local_data_count_{};
  std::vector<uint8_t> local_data_;
  std::vector<uint8_t> local_output_;
  std::vector<uint8_t> extended_data_;
};

}  // namespace otcheskov_s_gauss_filter_vert_split
