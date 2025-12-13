#include "golovanov_d_matrix_max_elem//mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "golovanov_d_matrix_max_elem//common/include/common.hpp"

namespace golovanov_d_matrix_max_elem {

GolovanovDMatrixMaxElemMPI::GolovanovDMatrixMaxElemMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 1234;
}

bool GolovanovDMatrixMaxElemMPI::ValidationImpl() {
  int columns = std::get<0>(GetInput());
  int strokes = std::get<1>(GetInput());
  return (columns > 0) && (strokes > 0) && (static_cast<int>(std::get<2>(GetInput()).size()) == (strokes * columns)) &&
         (GetOutput() == 1234);
}

bool GolovanovDMatrixMaxElemMPI::PreProcessingImpl() {
  return true;
}

bool GolovanovDMatrixMaxElemMPI::RunImpl() {
  int rank = 0;
  int processes = 0;
  int n = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &processes);
  std::vector<double> elems;
  double answer = 0;
  if (rank == 0) {
    auto columns = std::get<0>(GetInput());
    auto strokes = std::get<1>(GetInput());
    elems = std::get<2>(GetInput());
    auto count = columns * strokes;

    n = count / processes;
    if (count % processes != 0) {
      for (int i = 0; i < processes - (count % processes); i++) {
        elems.push_back(elems[0]);
      }
      n++;
    }
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<double> work_vector(n);
  MPI_Scatter(elems.data(), n, MPI_DOUBLE, work_vector.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  double max = *std::ranges::max_element(work_vector);
  MPI_Allreduce(&max, &answer, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  GetOutput() = answer;
  return true;
}

bool GolovanovDMatrixMaxElemMPI::PostProcessingImpl() {
  return true;
}

}  // namespace golovanov_d_matrix_max_elem
