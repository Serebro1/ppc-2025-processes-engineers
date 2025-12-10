#include "otcheskov_s_linear_topology/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <numeric>
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

void HandleSourceProcess(const Message &msg, int direction, int next_hop, MPI_Comm cart_comm, Response &response) {
  if (msg.src == msg.dest) {
    response.delivered = true;
    response.received_data = msg.data;
    return;
  }

  if (next_hop == MPI_PROC_NULL) {
    return;
  }

  int data_size = static_cast<int>(msg.data.size());
  MPI_Send(&data_size, 1, MPI_INT, next_hop, 0, cart_comm);
  MPI_Send(msg.data.data(), data_size, MPI_INT, next_hop, 1, cart_comm);
}

void HandleIntermediateProcess(const Message &msg, int direction, int cart_rank, int left_neighbor, int right_neighbor,
                               MPI_Comm cart_comm, Response &response) {
  bool is_on_path = false;
  if (direction > 0) {
    is_on_path = (cart_rank > msg.src && cart_rank < msg.dest);
  } else if (direction < 0) {
    is_on_path = (cart_rank < msg.src && cart_rank > msg.dest);
  }

  if (!is_on_path) {
    return;
  }

  int prev_hop = (direction > 0) ? left_neighbor : right_neighbor;
  int next_hop = (direction > 0) ? right_neighbor : left_neighbor;

  int data_size;
  MPI_Status status;
  MPI_Recv(&data_size, 1, MPI_INT, prev_hop, 0, cart_comm, &status);

  std::vector<int> temp_data(data_size);
  MPI_Recv(temp_data.data(), data_size, MPI_INT, prev_hop, 1, cart_comm, &status);

  if (next_hop != MPI_PROC_NULL) {
    MPI_Send(&data_size, 1, MPI_INT, next_hop, 0, cart_comm);
    MPI_Send(temp_data.data(), data_size, MPI_INT, next_hop, 1, cart_comm);
  }
}

void HandleDestinationProcess(const Message &msg, int direction, int left_neighbor, int right_neighbor,
                              MPI_Comm cart_comm, Response &response) {
  response.original_src = msg.src;
  response.final_dest = msg.dest;
  response.delivered = true;

  int prev_hop = (direction > 0) ? left_neighbor : right_neighbor;
  int data_size;
  MPI_Status status;

  MPI_Recv(&data_size, 1, MPI_INT, prev_hop, 0, cart_comm, &status);
  response.received_data.resize(data_size);
  MPI_Recv(response.received_data.data(), data_size, MPI_INT, prev_hop, 1, cart_comm, &status);

  if (direction > 0) {
    for (int i = msg.src; i <= msg.dest; ++i) {
      response.path.push_back(i);
    }
  } else {
    for (int i = msg.src; i >= msg.dest; --i) {
      response.path.push_back(i);
    }
  }
  response.hops = static_cast<int>(response.path.size()) - 1;
}

Response OtcheskovSLinearTopologySEQ::SendMessageWithMPICart(const Message &msg) {
  Response response;
  response.original_src = msg.src;
  response.final_dest = msg.dest;
  response.delivered = false;
  response.hops = 0;

  MPI_Comm cart_comm;
  int dims = proc_num_;
  int periods = 0;
  int reorder = 0;
  MPI_Cart_create(MPI_COMM_WORLD, 1, &dims, &periods, reorder, &cart_comm);

  int cart_rank;
  MPI_Comm_rank(cart_comm, &cart_rank);

  int src_coord, dest_coord;
  MPI_Cart_coords(cart_comm, msg.src, 1, &src_coord);
  MPI_Cart_coords(cart_comm, msg.dest, 1, &dest_coord);

  int left_neighbor, right_neighbor;
  MPI_Cart_shift(cart_comm, 0, 1, &left_neighbor, &right_neighbor);

  int direction = (dest_coord > src_coord) ? 1 : -1;

  if (proc_rank_ == msg.src) {
    HandleSourceProcess(msg, direction, (direction > 0) ? right_neighbor : left_neighbor, cart_comm, response);
  } else if (proc_rank_ == msg.dest) {
    HandleDestinationProcess(msg, direction, left_neighbor, right_neighbor, cart_comm, response);
  } else {
    HandleIntermediateProcess(msg, direction, cart_rank, left_neighbor, right_neighbor, cart_comm, response);
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

  int data_size = static_cast<int>(GetInput().data.size());
  MPI_Bcast(&data_size, 1, MPI_INT, GetInput().src, MPI_COMM_WORLD);

  broadcast_msg.data.resize(data_size);
  MPI_Bcast(broadcast_msg.data.data(), data_size, MPI_INT, GetInput().src, MPI_COMM_WORLD);

  Response result = SendMessageWithMPICart(broadcast_msg);

  if (proc_rank_ == broadcast_msg.dest) {
    if (result.received_data.empty()) {
      result.delivered = false;
    }
  }

  int local_delivery = result.delivered ? 1 : 0;
  int global_delivery = 0;
  MPI_Reduce(&local_delivery, &global_delivery, 1, MPI_INT, MPI_SUM, broadcast_msg.dest, MPI_COMM_WORLD);

  if (proc_rank_ == broadcast_msg.dest && global_delivery == 0) {
    result.delivered = false;
  }

  GetOutput() = result;
  return true;
}

bool OtcheskovSLinearTopologySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace otcheskov_s_linear_topology
