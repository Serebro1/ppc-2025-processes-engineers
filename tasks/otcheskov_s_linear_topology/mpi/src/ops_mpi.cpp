#include "otcheskov_s_linear_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
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

void OtcheskovSLinearTopologyMPI::SendMessageMPI(int dest, const Message &msg, int tag) {
  const auto &[header, data] = msg;
  MPI_Send(&header, sizeof(MessageHeader), MPI_BYTE, dest, tag, MPI_COMM_WORLD);

  if (header.data_size > 0) {
    MPI_Send(data.data(), header.data_size, MPI_INT, dest, tag + kDataTagOffset, MPI_COMM_WORLD);
  }
}

Message OtcheskovSLinearTopologyMPI::RecvMessageMPI(int src, int tag) {
  MessageHeader header;
  MPI_Status status;
  MPI_Recv(&header, sizeof(MessageHeader), MPI_BYTE, src, tag, MPI_COMM_WORLD, &status);

  MessageData data;
  if (header.data_size > 0) {
    data.resize(header.data_size);
    MPI_Recv(data.data(), header.data_size, MPI_INT, src, tag + kDataTagOffset, MPI_COMM_WORLD, &status);
  }
  return {header, std::move(data)};
}

Message OtcheskovSLinearTopologyMPI::SendMessageLinear(const Message &msg) const {
  auto [header, data] = msg;
  header.delivered = 0;

  if (header.src == header.dest) {
    if (proc_rank_ == header.src) {
      header.delivered = 1;
    }
    MPI_Bcast(&header.delivered, 1, MPI_INT, header.src, MPI_COMM_WORLD);

    if (proc_rank_ != header.src) {
      data.clear();
    }
    return {header, std::move(data)};
  }

  const int direction = (header.dest > header.src) ? 1 : -1;
  Message current_msg = {header, data};
  const bool shouldParticipate = (direction > 0 && proc_rank_ >= header.src && proc_rank_ <= header.dest) ||
                                 (direction < 0 && proc_rank_ <= header.src && proc_rank_ >= header.dest);

  if (!shouldParticipate) {
    return {MessageHeader(), MessageData()};
  }

  const bool isSrc = (proc_rank_ == header.src);
  const bool isDest = (proc_rank_ == header.dest);

  const int prev = isSrc ? MPI_PROC_NULL : proc_rank_ - direction;
  const int next = isDest ? MPI_PROC_NULL : proc_rank_ + direction;

  if (!isSrc) {
    current_msg = RecvMessageMPI(prev, kMessageTag);
  } else {
    current_msg = {header, data};
  }

  if (!isDest) {
    SendMessageMPI(next, current_msg, kMessageTag);
  } else {
    current_msg.first.delivered = 1;
  }

  // пересылка подтверждения
  if (isDest) {
    SendMessageMPI(prev, current_msg, kConfirmTag);
  } else {
    Message confirmation = RecvMessageMPI(next, kConfirmTag);

    if (isSrc) {
      current_msg.first.delivered = confirmation.first.delivered;
    } else {
      SendMessageMPI(prev, confirmation, kConfirmTag);
      current_msg.second.clear();
    }
  }
  return current_msg;
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
    check_passed = result_msg.first.delivered != 0 && !result_msg.second.empty();
  } else if (proc_rank_ == dest) {
    check_passed = !result_msg.second.empty() && result_msg.first.delivered != 0;
  } else {
    check_passed = true;
  }

  GetOutput() = result_msg;
  return check_passed;
}

bool OtcheskovSLinearTopologyMPI::PostProcessingImpl() {
  return true;
}

}  // namespace otcheskov_s_linear_topology
