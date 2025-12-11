#include "klimenko_v_max_matrix_elems_val/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>

#include "klimenko_v_max_matrix_elems_val/common/include/common.hpp"

namespace klimenko_v_max_matrix_elems_val {

KlimenkoVMaxMatrixElemsValMPI::KlimenkoVMaxMatrixElemsValMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = 0;
}

bool KlimenkoVMaxMatrixElemsValMPI::ValidationImpl() {
  if (GetInput().empty() || GetInput()[0].empty()) {
    GetOutput() = 0;
  }
  return true;
}

bool KlimenkoVMaxMatrixElemsValMPI::PreProcessingImpl() {
  return GetOutput() == 0;
}

bool KlimenkoVMaxMatrixElemsValMPI::RunImpl() {
  const auto &matrix = GetInput();

  int rank = 0;
  int size = 1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(matrix.size());
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n == 0) {
    int result = 0;
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    GetOutput() = result;
    return true;
  }

  int global_max = INT_MIN;

  if (rank == 0) {
    int local_max = INT_MIN;
    for (const auto &row : matrix) {
      for (int v : row) {
        local_max = std::max(local_max, v);
      }
    }
    global_max = local_max;
  }

  MPI_Bcast(&global_max, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_max;

  return true;
}

bool KlimenkoVMaxMatrixElemsValMPI::PostProcessingImpl() {
  return true;
}

}  // namespace klimenko_v_max_matrix_elems_val
