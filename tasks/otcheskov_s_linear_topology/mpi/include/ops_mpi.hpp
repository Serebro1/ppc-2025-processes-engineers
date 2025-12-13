#pragma once

#include "otcheskov_s_linear_topology/common/include/common.hpp"
#include "task/include/task.hpp"

namespace otcheskov_s_linear_topology {

class OtcheskovSLinearTopologyMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit OtcheskovSLinearTopologyMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void SendMessageMPI(int dest, const Message &msg, int tag);
  static Message RecvMessageMPI(int src, int tag);
  Message SendMessageLinear(const Message &msg);

  int proc_rank_{};
  int proc_num_{};
};

}  // namespace otcheskov_s_linear_topology
