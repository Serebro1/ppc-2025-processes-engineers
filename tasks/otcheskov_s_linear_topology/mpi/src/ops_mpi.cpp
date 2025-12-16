#include "otcheskov_s_linear_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "otcheskov_s_linear_topology/common/include/common.hpp"

namespace otcheskov_s_linear_topology {

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
    is_valid = (header.dest >= 0 && header.dest < proc_num_ && !header.delivered && !data.empty() &&
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
    MPI_Send(data.data(), header.data_size, MPI_INT, dest, tag + 1, MPI_COMM_WORLD);
  }
}

Message OtcheskovSLinearTopologyMPI::RecvMessageMPI(int src, int tag) {
  MessageHeader header;
  MPI_Status status;
  MPI_Recv(&header, sizeof(MessageHeader), MPI_BYTE, src, tag, MPI_COMM_WORLD, &status);

  MessageData data;
  if (header.data_size > 0) {
    data.resize(header.data_size);
    MPI_Recv(data.data(), header.data_size, MPI_INT, src, tag + 1, MPI_COMM_WORLD, &status);
  }
  return {header, data};
}

Message OtcheskovSLinearTopologyMPI::SendMessageLinear(const Message &msg) {
  auto [header, data] = msg;
  header.delivered = false;

  if (header.src == header.dest) {
    if (proc_rank_ == header.src) {
      header.delivered = true;
    }
    MPI_Bcast(&header.delivered, 1, MPI_INT, header.src, MPI_COMM_WORLD);
    return {header, data};
  }

  int direction = (header.dest > header.src) ? 1 : -1;
  int start = std::min(header.src, header.dest);
  int end = std::max(header.src, header.dest);

  Message current_msg = {header, data};

  if (proc_rank_ == header.src) {
    int next = proc_rank_ + direction;
    SendMessageMPI(next, current_msg, 100);
  }

  if (proc_rank_ > start && proc_rank_ < end) {
    int prev = proc_rank_ - direction;
    Message received = RecvMessageMPI(prev, 100);

    int next = proc_rank_ + direction;
    SendMessageMPI(next, received, 100);
    current_msg = received;
  }

  if (proc_rank_ == header.dest) {
    int prev = proc_rank_ - direction;
    current_msg = RecvMessageMPI(prev, 100);

    current_msg.first.delivered = true;
    SendMessageMPI(prev, current_msg, 200);
  }

  // Процессы на обратном пути
  if (proc_rank_ > start && proc_rank_ < end) {
    int next = proc_rank_ + direction;
    Message confirmation = RecvMessageMPI(next, 200);

    int prev = proc_rank_ - direction;
    SendMessageMPI(prev, confirmation, 200);
    current_msg = confirmation;
    current_msg.second.clear();
    current_msg.second.shrink_to_fit();
  }

  if (proc_rank_ == header.src) {
    int next = proc_rank_ + direction;
    current_msg = RecvMessageMPI(next, 200);
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
  msg_header.delivered = false;
  msg_header.data_size = 0;

  MessageData data;
  if (proc_rank_ == src) {
    data = GetInput().second;
    msg_header.data_size = static_cast<int>(data.size());
  }

  Message result_msg = SendMessageLinear({msg_header, data});

  bool check_passed = false;
  if (proc_rank_ == src) {
    check_passed = result_msg.first.delivered && !result_msg.second.empty();
  } else if (proc_rank_ == dest) {
    check_passed = !result_msg.second.empty() && result_msg.first.delivered;
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
