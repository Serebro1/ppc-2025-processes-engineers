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
  int GetLocalIndex(int row, int local_col, int channel, int width);
  void BroadcastImgMetadata();

  void DistributeData();
  int CalcColsForProcs(int proc_id, int total_width, int num_procs);
  int CalcStartCol(int proc_id, int total_width, int num_procs);
  void PrepareAndScatterData();
  void ReceiveLocalData();

  void ExchangeBoundaryColumns();
  void ExtractBoundaryColumns(std::vector<uint8_t> &left_column, std::vector<uint8_t> &right_column);
  void ExchangeColumnsWithNeighbors(const std::vector<uint8_t> &left_column, const std::vector<uint8_t> &right_column,
                                    std::vector<uint8_t> &received_left, std::vector<uint8_t> &received_right,
                                    int left_proc, int right_proc);
  void CreateExtendedData(const std::vector<uint8_t> &received_left, const std::vector<uint8_t> &received_right);

  void ApplyGaussianFilter();
  bool IsBoundaryPixel(int row, int col);
  void HandleBoundaryPixel(int row, int local_j, int global_j, int channel);
  void ApplyGaussianKernel(int row, int local_j, int channel, int extended_width);

  void CollectResults();
  void GatherResultsFromAllProcesses();
  void CopyLocalResultsToOutput();
  void ReceiveAndStoreProcessResults(int proc_rank);
  void SendLocalResults();

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
