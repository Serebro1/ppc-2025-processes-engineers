#include "kopilov_d_sum_val_col_mat/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "kopilov_d_sum_val_col_mat/common/include/common.hpp"

namespace kopilov_d_sum_val_col_mat {

KopilovDSumValColMatMPI::KopilovDSumValColMatMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool KopilovDSumValColMatMPI::ValidationImpl() {
  return true;
}

bool KopilovDSumValColMatMPI::PreProcessingImpl() {
  return true;
}

bool KopilovDSumValColMatMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int rows = 0;
  int cols = 0;
  const double *send_buffer_ptr = nullptr;

  if (world_rank == 0) {
    rows = GetInput().rows;
    cols = GetInput().cols;
    const std::size_t expected_size = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    if (GetInput().data.size() != expected_size) {
      return false;
    }
    send_buffer_ptr = GetInput().data.data();
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rows == 0 || cols == 0) {
    GetOutput().col_sum.clear();
    return true;
  }

  // compute rows per process (by rows, not by elements) to avoid fractional rows
  std::vector<int> rows_per_rank(static_cast<std::size_t>(world_size), 0);
  const int base_rows = rows / world_size;
  const int remainder_rows = rows % world_size;
  for (int pid = 0; pid < world_size; ++pid) {
    rows_per_rank[static_cast<std::size_t>(pid)] = base_rows + (pid < remainder_rows ? 1 : 0);
  }

  // send_counts/displs are in number of doubles (elements)
  std::vector<int> send_counts(static_cast<std::size_t>(world_size), 0);
  std::vector<int> displacements(static_cast<std::size_t>(world_size), 0);
  int offset = 0;
  for (int pid = 0; pid < world_size; ++pid) {
    send_counts[static_cast<std::size_t>(pid)] = rows_per_rank[static_cast<std::size_t>(pid)] * cols;
    displacements[static_cast<std::size_t>(pid)] = offset;
    offset += send_counts[static_cast<std::size_t>(pid)];
  }

  const int recv_count = send_counts[static_cast<std::size_t>(world_rank)];
  std::vector<double> recv_buffer(static_cast<std::size_t>(recv_count), 0.0);

  MPI_Scatterv(send_buffer_ptr, send_counts.data(), displacements.data(), MPI_DOUBLE, recv_buffer.data(), recv_count,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  const int local_rows = (cols == 0) ? 0 : static_cast<int>(recv_buffer.size()) / cols;
  std::vector<double> local_col_sum(static_cast<std::size_t>(cols), 0.0);

  for (int row = 0; row < local_rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      local_col_sum[static_cast<std::size_t>(col)] +=
          recv_buffer[(static_cast<std::size_t>(row) * static_cast<std::size_t>(cols)) + static_cast<std::size_t>(col)];
    }
  }

  std::vector<double> global_col_sum(static_cast<std::size_t>(cols), 0.0);
  MPI_Reduce(local_col_sum.data(), global_col_sum.data(), cols, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // Broadcast result to all processes so tests can validate on any rank
  MPI_Bcast(global_col_sum.data(), cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput().col_sum = std::move(global_col_sum);

  return true;
}

bool KopilovDSumValColMatMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kopilov_d_sum_val_col_mat
