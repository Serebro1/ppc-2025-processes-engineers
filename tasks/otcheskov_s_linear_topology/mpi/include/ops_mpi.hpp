#pragma once

#include "otcheskov_s_linear_topology/common/include/common.hpp"
#include "task/include/task.hpp"

namespace otcheskov_s_linear_topology {

class OtcheskovSLinearTopologyMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit OtcheskovSLinearTopologyMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void SendData(int dest, const std::vector<int> &data, int tag);
  bool RecvData(int src, std::vector<int> &data, int tag);
  Response SendMessageLinear(const Message &msg);

  int proc_rank_{};
  int proc_num_{};
};

}  // namespace otcheskov_s_linear_topology
