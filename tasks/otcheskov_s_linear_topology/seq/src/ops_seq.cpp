#include "otcheskov_s_linear_topology/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <cmath>
#include <vector>

#include "otcheskov_s_linear_topology/common/include/common.hpp"
#include "util/include/util.hpp"

namespace otcheskov_s_linear_topology {

OtcheskovSLinearTopologySEQ::OtcheskovSLinearTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num_);

  if (in.src == proc_rank_) {
    GetInput() = in;
  }

  GetOutput() = Response{false, 0, {}, {}, -1, -1};
}

bool OtcheskovSLinearTopologySEQ::ValidationImpl() {
  bool is_valid = true;

  if (proc_rank_ == GetInput().src) {
    is_valid = (GetInput().src >= 0 && GetInput().src < proc_num_ && GetInput().dest >= 0 &&
                GetInput().dest < proc_num_ && !GetInput().data.empty());
  }

  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, GetInput().src, MPI_COMM_WORLD);
  return is_valid;
}

bool OtcheskovSLinearTopologySEQ::PreProcessingImpl() {
  return true;
}

void OtcheskovSLinearTopologySEQ::SendData(int next_hop, const std::vector<int> &data, MPI_Comm comm) {
  int data_size = static_cast<int>(data.size());
  MPI_Send(&data_size, 1, MPI_INT, next_hop, 0, comm);
  MPI_Send(data.data(), data_size, MPI_INT, next_hop, 1, comm);
}

std::vector<int> OtcheskovSLinearTopologySEQ::RecvData(int prev_hop, MPI_Comm comm) {
  int data_size;
  MPI_Status status;
  MPI_Recv(&data_size, 1, MPI_INT, prev_hop, 0, comm, &status);

  std::vector<int> data(data_size);
  MPI_Recv(data.data(), data_size, MPI_INT, prev_hop, 1, comm, &status);
  return data;
}

Response OtcheskovSLinearTopologySEQ::SendMessageWithMPICart(const Message &msg) {
  Response response;
  response.orig_src = msg.src;
  response.final_dest = msg.dest;

  if (msg.src == msg.dest) {
    response.delivered = true;
    response.received_data = msg.data;
    response.path = {msg.src};
    response.hops = 0;
    return response;
  }

  MPI_Comm cart_comm;
  int dims = proc_num_;
  int periods = 0;
  MPI_Cart_create(MPI_COMM_WORLD, 1, &dims, &periods, 0, &cart_comm);

  int cart_rank;
  MPI_Cart_rank(cart_comm, &cart_rank);

  int left_neighbor, right_neighbor;
  MPI_Cart_shift(cart_comm, 0, 1, &left_neighbor, &right_neighbor);

  if (proc_rank_ == msg.src) {
    int direction = (msg.dest > msg.src) ? 1 : -1;
    int next_hop = (direction > 0) ? right_neighbor : left_neighbor;

    if (next_hop != MPI_PROC_NULL) {
      SendData(next_hop, msg.data, cart_comm);
    }
  } else if (proc_rank_ == msg.dest) {
    int direction = (msg.dest > msg.src) ? 1 : -1;
    int prev_hop = (direction > 0) ? left_neighbor : right_neighbor;

    response.received_data = RecvData(prev_hop, cart_comm);
    response.delivered = !response.received_data.empty();

    if (response.delivered) {
      int step = (msg.dest > msg.src) ? 1 : -1;
      for (int i = msg.src; i != msg.dest + step; i += step) {
        response.path.push_back(i);
      }
      response.hops = static_cast<int>(response.path.size()) - 1;
    }
  } else {
    int min_rank = std::min(msg.src, msg.dest);
    int max_rank = std::max(msg.src, msg.dest);

    if (proc_rank_ > min_rank && proc_rank_ < max_rank) {
      int direction = (msg.dest > msg.src) ? 1 : -1;
      int prev_hop = (direction > 0) ? left_neighbor : right_neighbor;
      int next_hop = (direction > 0) ? right_neighbor : left_neighbor;

      auto data = RecvData(prev_hop, cart_comm);

      if (next_hop != MPI_PROC_NULL) {
        SendData(next_hop, data, cart_comm);
      }
    }
  }

  MPI_Comm_free(&cart_comm);
  return response;
}

bool OtcheskovSLinearTopologySEQ::RunImpl() {
  Message broadcast_msg;

  if (proc_rank_ == GetInput().src) {
    broadcast_msg = GetInput();
  }

  MPI_Bcast(&broadcast_msg.src, 1, MPI_INT, GetInput().src, MPI_COMM_WORLD);
  MPI_Bcast(&broadcast_msg.dest, 1, MPI_INT, GetInput().src, MPI_COMM_WORLD);

  Response result = SendMessageWithMPICart(broadcast_msg);

  if (proc_rank_ == broadcast_msg.dest && result.received_data.empty()) {
    result.delivered = false;
  }

  GetOutput() = result;
  return true;
}

bool OtcheskovSLinearTopologySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace otcheskov_s_linear_topology
