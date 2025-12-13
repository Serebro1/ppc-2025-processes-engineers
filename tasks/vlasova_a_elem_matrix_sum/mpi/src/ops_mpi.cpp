#include "vlasova_a_elem_matrix_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <vector>

#include "vlasova_a_elem_matrix_sum/common/include/common.hpp"

namespace vlasova_a_elem_matrix_sum {

VlasovaAElemMatrixSumMPI::VlasovaAElemMatrixSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetOutput() = {};

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> empty_data;
  InType processed_input = std::make_tuple(empty_data, std::get<1>(in), std::get<2>(in));

  if (rank == 0) {
    GetInput() = in;
  } else {
    GetInput() = processed_input;
  }
}

bool VlasovaAElemMatrixSumMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank != 0) {
    return true;
  }

  int rows = std::get<1>(GetInput());
  int cols = std::get<2>(GetInput());
  const auto &matrix_data = std::get<0>(GetInput());

  return matrix_data.size() == static_cast<size_t>(rows) * static_cast<size_t>(cols) && rows > 0 && cols > 0;
}

bool VlasovaAElemMatrixSumMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    int rows = std::get<1>(GetInput());
    GetOutput().resize(rows, 0);
  }
  return true;
}

bool VlasovaAElemMatrixSumMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int> dimensions(2, 0);
  if (rank == 0) {
    dimensions[0] = std::get<1>(GetInput());
    dimensions[1] = std::get<2>(GetInput());
  }

  MPI_Bcast(dimensions.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
  const int total_rows = dimensions[0];
  const int total_cols = dimensions[1];

  if (total_rows == 0 || total_cols == 0) {
    return true;
  }

  const int base_rows_per_process = total_rows / size;
  const int remaining_rows = total_rows % size;

  const int local_row_count = base_rows_per_process + (rank < remaining_rows ? 1 : 0);
  const int row_offset = (rank * base_rows_per_process) + std::min(rank, remaining_rows);

  std::vector<int> local_matrix_data(static_cast<size_t>(local_row_count) * static_cast<size_t>(total_cols));

  std::vector<int> send_counts(size);
  std::vector<int> displacements(size);

  int current_displacement = 0;
  for (int proc = 0; proc < size; ++proc) {
    int proc_rows = base_rows_per_process + (proc < remaining_rows ? 1 : 0);
    send_counts[proc] = proc_rows * total_cols;
    displacements[proc] = current_displacement;
    current_displacement += send_counts[proc];
  }

  const auto &global_data = rank == 0 ? std::get<0>(GetInput()) : std::vector<int>();

  MPI_Scatterv(global_data.data(), send_counts.data(), displacements.data(), MPI_INT, local_matrix_data.data(),
               local_row_count * total_cols, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_sums(total_rows, 0);
  for (int i = 0; i < local_row_count; ++i) {
    int global_row_index = row_offset + i;
    for (int j = 0; j < total_cols; ++j) {
      local_sums[global_row_index] += local_matrix_data[(i * total_cols) + j];
    }
  }

  std::vector<int> final_sums(total_rows);
  MPI_Allreduce(local_sums.data(), final_sums.data(), total_rows, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = final_sums;
  return true;
}

bool VlasovaAElemMatrixSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace vlasova_a_elem_matrix_sum
