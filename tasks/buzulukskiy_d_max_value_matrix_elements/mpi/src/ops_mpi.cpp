#include "buzulukskiy_d_max_value_matrix_elements/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "buzulukskiy_d_max_value_matrix_elements/common/include/common.hpp"

namespace buzulukskiy_d_max_value_matrix_elements {

BuzulukskiyDMaxValueMatrixElementsMPI::BuzulukskiyDMaxValueMatrixElementsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool BuzulukskiyDMaxValueMatrixElementsMPI::ValidationImpl() {
  const Matrix &inputdata = GetInput();
  const int rows = inputdata.rows;
  const int columns = inputdata.columns;
  const std::vector<int> &matrix = inputdata.data;

  return !matrix.empty() && rows > 0 && columns > 0 &&
         matrix.size() == static_cast<size_t>(rows) * static_cast<size_t>(columns);
}

bool BuzulukskiyDMaxValueMatrixElementsMPI::PreProcessingImpl() {
  return true;
}

bool BuzulukskiyDMaxValueMatrixElementsMPI::RunImpl() {
  const Matrix &inputdata = GetInput();
  const int rows = inputdata.rows;
  const int columns = inputdata.columns;
  const std::vector<int> &matrix = inputdata.data;

  if (matrix.empty()) {
    GetOutput() = 0;
    return true;
  }

  int rank = 0;
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int rows_at_one_process = rows / size;
  const int remaining_rows = rows % size;
  std::vector<int> counts_per_process(size, rows_at_one_process * columns);
  std::vector<int> displacements(size, 0);

  if (remaining_rows != 0) {
    for (int process_index = 0; process_index < remaining_rows; ++process_index) {
      counts_per_process[process_index] += columns;
    }
  }
  for (int process_index = 1; process_index < size; ++process_index) {
    displacements[process_index] = displacements[process_index - 1] + counts_per_process[process_index - 1];
  }

  std::vector<int> recvbuf(counts_per_process[rank]);
  MPI_Scatterv(matrix.data(), counts_per_process.data(), displacements.data(), MPI_INT, recvbuf.data(),
               counts_per_process[rank], MPI_INT, 0, MPI_COMM_WORLD);

  int local_max = std::numeric_limits<int>::min();
  if (!recvbuf.empty()) {
    local_max = recvbuf[0];
    for (size_t index = 1; index < recvbuf.size(); ++index) {
      local_max = std::max(local_max, recvbuf[index]);
    }
  }

  int global_max = 0;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  GetOutput() = global_max;
  return true;
}

bool BuzulukskiyDMaxValueMatrixElementsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace buzulukskiy_d_max_value_matrix_elements
