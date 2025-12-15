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

  int GetIndex(int row, int col, int channel) const;
  void distributeData(std::vector<int> &local_data);
  void exchangeBoundaryRows(std::vector<int> &local_data);
  void applyGaussianFilter(const std::vector<int> &ext_data, std::vector<int> &local_output);
  void collectResults(const std::vector<int> &local_output);

  int proc_rank_{};
  int proc_num_{};
  int local_height_{};
  int start_row_{};
  int local_data_count_{};
  std::vector<int> local_input_;
};

}  // namespace otcheskov_s_gauss_filter_vert_split
