#include "otcheskov_s_linear_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

namespace otcheskov_s_linear_topology {

OtcheskovSLinearTopologyMPI::OtcheskovSLinearTopologyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num_);

  GetInput() = in;
  if (proc_rank_ != in.src) {
    GetInput().data.clear();
  }
  GetOutput() = {};
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
  MPI_Send(&msg.delivered, 1, MPI_C_BOOL, dest, tag, MPI_COMM_WORLD);
  MPI_Send(&msg.src, 1, MPI_INT, dest, tag + 1, MPI_COMM_WORLD);
  MPI_Send(&msg.dest, 1, MPI_INT, dest, tag + 2, MPI_COMM_WORLD);

  int data_size = static_cast<int>(msg.data.size());
  MPI_Send(&data_size, 1, MPI_INT, dest, tag + 3, MPI_COMM_WORLD);
  if (data_size > 0) {
    MPI_Send(msg.data.data(), data_size, MPI_INT, dest, tag + 4, MPI_COMM_WORLD);
  }
}

Message OtcheskovSLinearTopologyMPI::RecvMessageMPI(int src, int tag) {
  Message msg;
  MPI_Status status;
  MPI_Recv(&msg.delivered, 1, MPI_C_BOOL, src, tag, MPI_COMM_WORLD, &status);
  MPI_Recv(&msg.src, 1, MPI_INT, src, tag + 1, MPI_COMM_WORLD, &status);
  MPI_Recv(&msg.dest, 1, MPI_INT, src, tag + 2, MPI_COMM_WORLD, &status);

  int data_size;
  MPI_Recv(&data_size, 1, MPI_INT, src, tag + 3, MPI_COMM_WORLD, &status);
  if (data_size > 0) {
    msg.data.resize(data_size);
    MPI_Recv(msg.data.data(), data_size, MPI_INT, src, tag + 4, MPI_COMM_WORLD, &status);
  } else {
    msg.data.clear();
  }

  return msg;
}

Message OtcheskovSLinearTopologyMPI::SendMessageLinear(const Message &msg) {
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
  } else {
    result.delivered = true;  // процессы, не участвующие в передаче сообщения
  }

  // Источник получает подтверждение
  if (proc_rank_ == msg.src) {
    int next = proc_rank_ + direction;
    result = RecvMessageMPI(next, 200);
  }

  return result;
}

bool OtcheskovSLinearTopologyMPI::RunImpl() {
  int src, dest;
  src = GetInput().src;
  dest = GetInput().dest;
  if (src < 0 || src >= proc_num_) {
    return false;
  }

  Message msg;
  msg.src = src;
  msg.dest = dest;
  msg.delivered = false;

  if (proc_rank_ == src) {
    msg.data = GetInput().data;
  } else {
    msg.data.clear();
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
