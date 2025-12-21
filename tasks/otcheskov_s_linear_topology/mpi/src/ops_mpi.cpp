#include "otcheskov_s_linear_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "otcheskov_s_linear_topology/common/include/common.hpp"

namespace otcheskov_s_linear_topology {

namespace {
constexpr int kMessageTag = 100;
constexpr int kConfirmTag = 200;
constexpr int kDataTagOffset = 1;
}  // namespace

OtcheskovSLinearTopologyMPI::OtcheskovSLinearTopologyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num_);

  GetInput() = in;
  if (proc_rank_ != in.first.src) {
    GetInput().second.clear();
    GetInput().second.shrink_to_fit();
  }
}

bool OtcheskovSLinearTopologyMPI::ValidationImpl() {
  const auto &[header, data] = GetInput();
  if (header.src < 0 || header.src >= proc_num_) {
    return false;
  }

  bool is_valid = false;

  if (proc_rank_ == header.src) {
    is_valid = (header.dest >= 0 && header.dest < proc_num_ && header.delivered == 0 && !data.empty() &&
                static_cast<size_t>(header.data_size) == data.size());
  }
  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, GetInput().first.src, MPI_COMM_WORLD);
  return is_valid;
}

bool OtcheskovSLinearTopologyMPI::PreProcessingImpl() {
  return true;
}

Message OtcheskovSLinearTopologyMPI::ForwardMessageToDest(const Message &initial_msg, int prev, int next, bool is_src,
                                                          bool is_dest) {
  Message current_msg;
  auto &[header, data] = current_msg;
  if (!is_src) {
    MPI_Recv(&header, sizeof(MessageHeader), MPI_BYTE, prev, kMessageTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    data.resize(header.data_size);
    MPI_Recv(data.data(), header.data_size, MPI_INT, prev, kMessageTag + kDataTagOffset, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  } else {
    current_msg = initial_msg;
  }

  if (!is_dest) {
    MPI_Send(&header, sizeof(MessageHeader), MPI_BYTE, next, kMessageTag, MPI_COMM_WORLD);
    MPI_Send(data.data(), header.data_size, MPI_INT, next, kMessageTag + kDataTagOffset, MPI_COMM_WORLD);
    if (!is_src) {
      data.clear();
      data.shrink_to_fit();
    }
  } else {
    header.delivered = 1;
  }
  return current_msg;
}

Message OtcheskovSLinearTopologyMPI::HandleConfirmToSource(Message &current_msg, int prev, int next, bool is_src,
                                                           bool is_dest) {
  auto &confirm_header = current_msg.first;
  if (is_dest) {
    MPI_Send(&confirm_header, sizeof(MessageHeader), MPI_BYTE, prev, kConfirmTag, MPI_COMM_WORLD);
  } else {
    MPI_Recv(&confirm_header, sizeof(MessageHeader), MPI_BYTE, next, kConfirmTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (!is_src) {
      MPI_Send(&confirm_header, sizeof(MessageHeader), MPI_BYTE, prev, kConfirmTag, MPI_COMM_WORLD);
    }
  }
  return current_msg;
}

Message OtcheskovSLinearTopologyMPI::SendMessageLinear(const Message &msg) const {
  auto [header, data] = msg;
  header.delivered = 0;

  if (header.src == header.dest) {
    header.delivered = 1;
    return {header, std::move(data)};
  }

  const int direction = (header.dest > header.src) ? 1 : -1;
  const bool should_participate = (direction > 0 && proc_rank_ >= header.src && proc_rank_ <= header.dest) ||
                                  (direction < 0 && proc_rank_ <= header.src && proc_rank_ >= header.dest);

  if (!should_participate) {
    return {MessageHeader(), MessageData()};
  }

  const bool is_src = (proc_rank_ == header.src);
  const bool is_dest = (proc_rank_ == header.dest);

  const int prev = is_src ? MPI_PROC_NULL : proc_rank_ - direction;
  const int next = is_dest ? MPI_PROC_NULL : proc_rank_ + direction;

  Message current_msg = ForwardMessageToDest({header, std::move(data)}, prev, next, is_src, is_dest);
  // пересылка подтверждения
  return HandleConfirmToSource(current_msg, prev, next, is_src, is_dest);
}

bool OtcheskovSLinearTopologyMPI::RunImpl() {
  const auto &in_header = GetInput().first;
  const int src = in_header.src;
  const int dest = in_header.dest;

  if (src < 0 || src >= proc_num_) {
    return false;
  }

  MessageHeader msg_header;
  msg_header.src = src;
  msg_header.dest = dest;
  msg_header.delivered = 0;
  msg_header.data_size = 0;

  MessageData data;
  if (proc_rank_ == src) {
    data = GetInput().second;
    msg_header.data_size = static_cast<int>(data.size());
  }

  Message result_msg = SendMessageLinear({msg_header, data});

  bool check_passed = false;
  if (proc_rank_ == src) {
    check_passed = result_msg.first.delivered != 0;
  } else if (proc_rank_ == dest) {
    check_passed = !result_msg.second.empty();
  } else {
    check_passed = true;
  }
  GetOutput() = result_msg;
  MPI_Barrier(MPI_COMM_WORLD);
  return check_passed;
}

bool OtcheskovSLinearTopologyMPI::PostProcessingImpl() {
  return true;
}

}  // namespace otcheskov_s_linear_topology
