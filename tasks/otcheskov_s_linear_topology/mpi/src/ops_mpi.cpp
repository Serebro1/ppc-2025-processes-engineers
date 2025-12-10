#include "otcheskov_s_linear_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <vector>

#include "otcheskov_s_linear_topology/common/include/common.hpp"

namespace otcheskov_s_linear_topology {

OtcheskovSLinearTopologyMPI::OtcheskovSLinearTopologyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num_);

  if (proc_rank_ == in.src) {
    GetInput() = in;
  }

  GetOutput() = Response{false, 0, {}, {}, -1, -1};
}

bool OtcheskovSLinearTopologyMPI::ValidationImpl() {
  bool is_valid = true;

  if (proc_rank_ == GetInput().src) {
    is_valid = (GetInput().src >= 0 && GetInput().src < proc_num_ && GetInput().dest >= 0 &&
                GetInput().dest < proc_num_ && !GetInput().data.empty());
  }

  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, GetInput().src, MPI_COMM_WORLD);
  return is_valid;
}

bool OtcheskovSLinearTopologyMPI::PreProcessingImpl() {
  return true;
}

void HandleSourceProcess(const Message &msg, int next_hop, Response &response) {
  const auto &data = msg.data;

  if (msg.src == msg.dest) {
    response.delivered = true;
    response.received_data = data;
    return;
  }

  if (next_hop == MPI_PROC_NULL) {
    return;
  }

  int data_size = static_cast<int>(data.size());
  MPI_Send(&data_size, 1, MPI_INT, next_hop, 0, MPI_COMM_WORLD);
  MPI_Send(data.data(), data_size, MPI_INT, next_hop, 1, MPI_COMM_WORLD);
}

void HandleIntermediateProcess(int prev_hop, int next_hop, Response &response) {
  if (prev_hop == MPI_PROC_NULL || next_hop == MPI_PROC_NULL) {
    return;
  }

  MPI_Status status;

  int data_size;
  MPI_Recv(&data_size, 1, MPI_INT, prev_hop, 0, MPI_COMM_WORLD, &status);

  std::vector<int> data(data_size);
  MPI_Recv(data.data(), data_size, MPI_INT, prev_hop, 1, MPI_COMM_WORLD, &status);

  if (next_hop != MPI_PROC_NULL) {
    MPI_Send(&data_size, 1, MPI_INT, next_hop, 0, MPI_COMM_WORLD);
    MPI_Send(data.data(), data_size, MPI_INT, next_hop, 1, MPI_COMM_WORLD);
  }
}

void HandleDestinationProcess(int prev_hop, const Message &msg, Response &response) {
  MPI_Status status;

  if (msg.src != msg.dest) {
    int data_size;
    MPI_Recv(&data_size, 1, MPI_INT, prev_hop, 0, MPI_COMM_WORLD, &status);

    response.received_data.resize(data_size);
    MPI_Recv(response.received_data.data(), data_size, MPI_INT, prev_hop, 1, MPI_COMM_WORLD, &status);
  } else {
    response.received_data = msg.data;
  }

  response.path.clear();
  if (msg.src <= msg.dest) {
    for (int i = msg.src; i <= msg.dest; ++i) {
      response.path.push_back(i);
    }
  } else {
    for (int i = msg.src; i >= msg.dest; --i) {
      response.path.push_back(i);
    }
  }

  response.delivered = true;
  response.hops = static_cast<int>(response.path.size()) - 1;
}

Response OtcheskovSLinearTopologyMPI::SendMessage(const Message &msg) {
  Response response;
  response.original_src = msg.src;
  response.final_dest = msg.dest;
  response.delivered = false;
  response.hops = 0;

  int left_neighbor = (proc_rank_ > 0) ? proc_rank_ - 1 : MPI_PROC_NULL;
  int right_neighbor = (proc_rank_ < proc_num_ - 1) ? proc_rank_ + 1 : MPI_PROC_NULL;

  int direction = 0;
  if (msg.dest > msg.src) {
    direction = 1;
  } else if (msg.dest < msg.src) {
    direction = -1;
  }

  bool is_on_path = false;
  if (direction > 0) {
    is_on_path = (proc_rank_ >= msg.src && proc_rank_ <= msg.dest);
  } else if (direction < 0) {
    is_on_path = (proc_rank_ <= msg.src && proc_rank_ >= msg.dest);
  } else {
    is_on_path = (proc_rank_ == msg.src);
  }

  if (!is_on_path) {
    return response;

    if (proc_rank_ == msg.src) {
      int next_hop = (direction > 0) ? right_neighbor : left_neighbor;
      HandleSourceProcess(msg, next_hop, response);
    } else if (proc_rank_ == msg.dest) {
      int prev_hop = (direction > 0) ? left_neighbor : right_neighbor;
      HandleDestinationProcess(prev_hop, msg, response);
    } else {
      int prev_hop = (direction > 0) ? left_neighbor : right_neighbor;
      int next_hop = (direction > 0) ? right_neighbor : left_neighbor;
      HandleIntermediateProcess(prev_hop, next_hop, response);
    }

    return response;
  }

  bool OtcheskovSLinearTopologyMPI::RunImpl() {
    Message msg;

    if (proc_rank_ == GetInput().src) {
      msg = GetInput();
    }

    MPI_Bcast(&msg.src, 1, MPI_INT, GetInput().src, MPI_COMM_WORLD);
    MPI_Bcast(&msg.dest, 1, MPI_INT, GetInput().src, MPI_COMM_WORLD);

    int data_size = static_cast<int>(GetInput().data.size());
    MPI_Bcast(&data_size, 1, MPI_INT, GetInput().src, MPI_COMM_WORLD);

    msg.data.resize(data_size);
    MPI_Bcast(msg.data.data(), data_size, MPI_INT, GetInput().src, MPI_COMM_WORLD);

    Response local_response = SendMessage(msg);

    bool delivery_status = false;
    if (proc_rank_ == msg.dest) {
      delivery_status = local_response.delivered;

      if (delivery_status && local_response.received_data.empty()) {
        delivery_status = false;
      }
    }

    MPI_Bcast(&delivery_status, 1, MPI_C_BOOL, msg.dest, MPI_COMM_WORLD);

    if (delivery_status) {
      Response final_response;

      if (proc_rank_ == msg.dest) {
        final_response = local_response;

        int path_size = static_cast<int>(final_response.path.size());
        MPI_Bcast(&path_size, 1, MPI_INT, msg.dest, MPI_COMM_WORLD);
        MPI_Bcast(final_response.path.data(), path_size, MPI_INT, msg.dest, MPI_COMM_WORLD);

        int data_size = static_cast<int>(final_response.received_data.size());
        MPI_Bcast(&data_size, 1, MPI_INT, msg.dest, MPI_COMM_WORLD);
        MPI_Bcast(final_response.received_data.data(), data_size, MPI_INT, msg.dest, MPI_COMM_WORLD);

        MPI_Bcast(&final_response.hops, 1, MPI_INT, msg.dest, MPI_COMM_WORLD);
      } else {
        int path_size, data_size;
        MPI_Bcast(&path_size, 1, MPI_INT, msg.dest, MPI_COMM_WORLD);
        final_response.path.resize(path_size);
        MPI_Bcast(final_response.path.data(), path_size, MPI_INT, msg.dest, MPI_COMM_WORLD);

        MPI_Bcast(&data_size, 1, MPI_INT, msg.dest, MPI_COMM_WORLD);
        final_response.received_data.resize(data_size);
        MPI_Bcast(final_response.received_data.data(), data_size, MPI_INT, msg.dest, MPI_COMM_WORLD);

        MPI_Bcast(&final_response.hops, 1, MPI_INT, msg.dest, MPI_COMM_WORLD);

        final_response.original_src = msg.src;
        final_response.final_dest = msg.dest;
        final_response.delivered = true;
      }

      GetOutput() = final_response;
    } else {
      GetOutput().original_src = msg.src;
      GetOutput().final_dest = msg.dest;
      GetOutput().delivered = false;
    }

    return true;
  }

  bool OtcheskovSLinearTopologyMPI::PostProcessingImpl() {
    return true;
  }

}  // namespace otcheskov_s_linear_topology
