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

  [[nodiscard]] static Message ForwardMessageToDest(const Message &initial_msg, int prev, int next, bool is_src,
                                                    bool is_dest);
  [[nodiscard]] static Message HandleConfirmToSource(Message &current_msg, int prev, int next, bool is_src,
                                                     bool is_dest);
  [[nodiscard]] Message SendMessageLinear(const Message &msg) const;

  int proc_rank_{};
  int proc_num_{};
};

}  // namespace otcheskov_s_linear_topology
