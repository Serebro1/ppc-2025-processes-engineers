#include "tabalaev_a_elem_mat_min/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <utility>
#include <vector>

#include "tabalaev_a_elem_mat_min/common/include/common.hpp"

namespace tabalaev_a_elem_mat_min {

TabalaevAElemMatMinMPI::TabalaevAElemMatMinMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool TabalaevAElemMatMinMPI::ValidationImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int validation = 0;

  if (world_rank == 0) {
    auto &rows = std::get<0>(GetInput());
    auto &columns = std::get<1>(GetInput());
    auto &matrix = std::get<2>(GetInput());

    if ((rows != 0 && columns != 0) && (!matrix.empty()) && (matrix.size() == rows * columns)) {
      validation = 1;
    }
  }

  MPI_Bcast(&validation, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return validation != 0;
}

bool TabalaevAElemMatMinMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool TabalaevAElemMatMinMPI::RunImpl() {
  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::vector<int> matrix;
  std::vector<int> sendcounts(world_size);
  std::vector<int> displs(world_size);
  std::vector<int> local_matrix;

  if (world_rank == 0) {
    auto &input = GetInput();
    matrix = std::get<2>(input);

    size_t matrix_size = matrix.size();
    size_t part_size = matrix_size / world_size;
    size_t remainder = matrix_size % world_size;

    int offset = 0;
    for (size_t i = 0; std::cmp_less(i, world_size); ++i) {
      sendcounts[i] = static_cast<int>(part_size) + (i < remainder ? 1 : 0);
      displs[i] = offset;
      offset += sendcounts[i];
    }
  }

  MPI_Bcast(sendcounts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
  int local_size = sendcounts[world_rank];
  local_matrix.resize(local_size);

  MPI_Scatterv(world_rank == 0 ? matrix.data() : nullptr, sendcounts.data(), displs.data(), MPI_INT,
               local_matrix.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

  int local_minik = INT_MAX;
  for (int elem : local_matrix) {
    local_minik = std::min(local_minik, elem);
  }

  int global_minik = 0;
  MPI_Allreduce(&local_minik, &global_minik, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = global_minik;

  return true;
}

bool TabalaevAElemMatMinMPI::PostProcessingImpl() {
  return true;
}

}  // namespace tabalaev_a_elem_mat_min
