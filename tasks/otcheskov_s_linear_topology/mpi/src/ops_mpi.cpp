#include "otcheskov_s_linear_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <vector>

#include "otcheskov_s_linear_topology/common/include/common.hpp"
#include "util/include/util.hpp"

namespace otcheskov_s_linear_topology {

OtcheskovSLinearTopologyMPI::OtcheskovSLinearTopologyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num_);

  if (in.src == proc_rank_) {
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

void OtcheskovSLinearTopologyMPI::SendData(int dest, const std::vector<int> &data, int tag) {
  int data_size = static_cast<int>(data.size());
  MPI_Send(&data_size, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
  MPI_Send(data.data(), data_size, MPI_INT, dest, tag + 1, MPI_COMM_WORLD);
}

bool OtcheskovSLinearTopologyMPI::RecvData(int src, std::vector<int> &data, int tag) {
  MPI_Status status;
  int flag = 0;
  MPI_Iprobe(src, tag, MPI_COMM_WORLD, &flag, &status);

  if (!flag) {
    return false;
  }

  int data_size;
  MPI_Recv(&data_size, 1, MPI_INT, src, tag, MPI_COMM_WORLD, &status);

  data.resize(data_size);
  MPI_Recv(data.data(), data_size, MPI_INT, src, tag + 1, MPI_COMM_WORLD, &status);
  return true;
}

Response OtcheskovSLinearTopologyMPI::SendMessageLinear(const Message &msg) {
  Response response;
  response.orig_src = msg.src;
  response.final_dest = msg.dest;

  // Если отправитель и получатель совпадают
  if (msg.src == msg.dest) {
    response.delivered = true;
    response.received_data = msg.data;
    response.path = {msg.src};
    response.hops = 0;
    return response;
  }

  // Рассчитываем направление и следующий процесс
  int direction = (msg.dest > msg.src) ? 1 : -1;
  int next_hop = msg.src + direction;

  // Процесс-отправитель
  if (proc_rank_ == msg.src) {
    // Отправляем данные следующему процессу
    if (next_hop >= 0 && next_hop < proc_num_) {
      SendData(next_hop, msg.data, 100);
    }
  }

  // Промежуточные процессы
  if (proc_rank_ != msg.src && proc_rank_ != msg.dest) {
    // Проверяем, находимся ли мы на пути
    int min_rank = std::min(msg.src, msg.dest);
    int max_rank = std::max(msg.src, msg.dest);

    if (proc_rank_ > min_rank && proc_rank_ < max_rank) {
      std::vector<int> data;
      int prev_hop = proc_rank_ - direction;

      if (RecvData(prev_hop, data, 100)) {
        next_hop = proc_rank_ + direction;
        if (next_hop >= 0 && next_hop < proc_num_) {
          SendData(next_hop, data, 100);
        }
      }
    }
  }

  if (proc_rank_ == msg.dest) {
    int prev_hop = msg.dest - direction;

    if (RecvData(prev_hop, response.received_data, 100)) {
      response.delivered = true;

      int step = (msg.dest > msg.src) ? 1 : -1;
      for (int i = msg.src; i != msg.dest + step; i += step) {
        response.path.push_back(i);
      }
      response.hops = static_cast<int>(response.path.size()) - 1;
    } else {
      response.delivered = false;
    }
  }

  return response;
}

bool OtcheskovSLinearTopologyMPI::RunImpl() {
  int src = -1, dest = -1;

  if (proc_rank_ == GetInput().src) {
    src = GetInput().src;
    dest = GetInput().dest;
  }

  MPI_Bcast(&src, 1, MPI_INT, GetInput().src, MPI_COMM_WORLD);
  MPI_Bcast(&dest, 1, MPI_INT, GetInput().src, MPI_COMM_WORLD);

  Message msg;
  msg.src = src;
  msg.dest = dest;

  if (proc_rank_ == src) {
    msg.data = GetInput().data;
  } else {
    msg.data.clear();
  }

  // Выполняем передачу сообщения
  Response result = SendMessageLinear(msg);

  GetOutput() = result;
  return true;
}

bool OtcheskovSLinearTopologyMPI::PostProcessingImpl() {
  return true;
}

}  // namespace otcheskov_s_linear_topology
