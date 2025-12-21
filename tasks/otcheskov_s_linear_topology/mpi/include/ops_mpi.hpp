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

  static void SendData(int dest, int size, const MessageData &data, int tag);
  static MessageData RecvData(int src, int size, int tag);

  static void SendHeader(int dest, const MessageHeader &header, int tag);
  static MessageHeader RecvHeader(int src, int tag);

  [[nodiscard]] static Message ForwardMessageToDest(const Message &initial_msg, int prev, int next, bool is_src,
                                                    bool is_dest);
  [[nodiscard]] static Message HandleConfirmToSource(Message &current_msg, int prev, int next, bool is_src,
                                                     bool is_dest);
  [[nodiscard]] Message SendMessageLinear(const Message &msg) const;

  int proc_rank_{};
  int proc_num_{};
};

}  // namespace otcheskov_s_linear_topology
