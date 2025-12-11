#pragma once

#include <mpi.h>

#include "otcheskov_s_linear_topology/common/include/common.hpp"
#include "task/include/task.hpp"

namespace otcheskov_s_linear_topology {

class OtcheskovSLinearTopologySEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit OtcheskovSLinearTopologySEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void SendData(int next_hop, const std::vector<int> &data, MPI_Comm comm);
  std::vector<int> RecvData(int prev_hop, MPI_Comm comm);
  Response SendMessageWithMPICart(const Message &msg);

  int proc_rank_{};
  int proc_num_{};
};

}  // namespace otcheskov_s_linear_topology
