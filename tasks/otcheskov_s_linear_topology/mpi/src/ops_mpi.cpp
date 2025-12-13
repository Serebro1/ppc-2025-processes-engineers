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
  if (GetInput().src < 0 || GetInput().src >= proc_num_) {
    return false;
  }

  bool is_valid = false;

  if (proc_rank_ == GetInput().src) {
    is_valid =
        (GetInput().dest >= 0 && GetInput().dest < proc_num_ && !GetInput().data.empty() && !GetInput().delivered);
  }
  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, GetInput().src, MPI_COMM_WORLD);
  return is_valid;
}

bool OtcheskovSLinearTopologyMPI::PreProcessingImpl() {
  return true;
}

void OtcheskovSLinearTopologyMPI::SendMessageMPI(int dest, const Message &msg, int tag) {
  MessageHeader header;
  header.delivered = msg.delivered;
  header.src = msg.src;
  header.dest = msg.dest;
  header.data_size = static_cast<int>(msg.data.size());
  MPI_Send(&header, sizeof(MessageHeader), MPI_BYTE, dest, tag, MPI_COMM_WORLD);

  if (header.data_size > 0) {
    MPI_Send(msg.data.data(), header.data_size, MPI_INT, dest, tag + 1, MPI_COMM_WORLD);
  }
}

Message OtcheskovSLinearTopologyMPI::RecvMessageMPI(int src, int tag) {
  MessageHeader header;
  MPI_Status status;
  MPI_Recv(&header, sizeof(MessageHeader), MPI_BYTE, src, tag, MPI_COMM_WORLD, &status);

  Message msg;
  msg.delivered = header.delivered;
  msg.src = header.src;
  msg.dest = header.dest;
  if (header.data_size > 0) {
    msg.data.resize(header.data_size);
    MPI_Recv(msg.data.data(), header.data_size, MPI_INT, src, tag + 1, MPI_COMM_WORLD, &status);
  }
  return msg;
}

Message OtcheskovSLinearTopologyMPI::SendMessageLinear(const Message &msg) const {
  Message result = msg;
  result.delivered = false;

  if (msg.src == msg.dest) {
    if (proc_rank_ == msg.src) {
      result.delivered = true;
    }
    MPI_Bcast(&result.delivered, 1, MPI_C_BOOL, msg.src, MPI_COMM_WORLD);
    return result;
  }

  int direction = (msg.dest > msg.src) ? 1 : -1;
  int start = std::min(msg.src, msg.dest);
  int end = std::max(msg.src, msg.dest);

  if (proc_rank_ == msg.src) {
    int next = proc_rank_ + direction;
    SendMessageMPI(next, result, 100);
  }

  if (proc_rank_ > start && proc_rank_ < end) {
    int prev = proc_rank_ - direction;
    Message received = RecvMessageMPI(prev, 100);

    int next = proc_rank_ + direction;
    SendMessageMPI(next, received, 100);
  }

  if (proc_rank_ == msg.dest) {
    int prev = proc_rank_ - direction;
    result = RecvMessageMPI(prev, 100);

    result.delivered = true;
    SendMessageMPI(prev, result, 200);
  }

  // Процессы на обратном пути
  if (proc_rank_ > start && proc_rank_ < end) {
    int next = proc_rank_ + direction;
    Message confirmation = RecvMessageMPI(next, 200);

    int prev = proc_rank_ - direction;
    SendMessageMPI(prev, confirmation, 200);
  }

  if (proc_rank_ == msg.src) {
    int next = proc_rank_ + direction;
    result = RecvMessageMPI(next, 200);
  }

  return result;
}

bool OtcheskovSLinearTopologyMPI::RunImpl() {
  const int src = GetInput().src;
  const int dest = GetInput().dest;

  if (src < 0 || src >= proc_num_) {
    return false;
  }

  Message msg;
  msg.src = src;
  msg.dest = dest;
  msg.delivered = false;
  msg.data = {};
  if (proc_rank_ == src) {
    msg.data = GetInput().data;
  }

  Message result_msg = SendMessageLinear(msg);

  bool check_passed = false;
  if (proc_rank_ == src) {
    check_passed = result_msg.delivered && !result_msg.data.empty();
  } else if (proc_rank_ == dest) {
    check_passed = !result_msg.data.empty() && result_msg.delivered;
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
