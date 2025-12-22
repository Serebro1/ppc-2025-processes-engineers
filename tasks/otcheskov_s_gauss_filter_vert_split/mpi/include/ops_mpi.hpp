#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
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

  void DistributeData();
  [[nodiscard]] std::pair<std::vector<int>, std::vector<int>> GetCountsAndDisplacements(const size_t &height,
                                                                                        const size_t &width,
                                                                                        const size_t &channels) const;

  void ExchangeBoundaryColumns();

  void ApplyGaussianFilter();
  uint8_t ProcessPixel(const size_t &row, const size_t &local_col, const size_t &ch, const size_t &height,
                       const size_t &channels);
  void CollectResults();

  bool is_valid_{};
  size_t proc_rank_{};
  size_t proc_num_{};
  size_t active_procs_{};

  size_t local_width_{};
  size_t start_col_{};
  size_t local_data_count_{};

  std::vector<uint8_t> local_data_;
  std::vector<uint8_t> extended_data_;
  std::vector<uint8_t> local_output_;
};

}  // namespace otcheskov_s_gauss_filter_vert_split
