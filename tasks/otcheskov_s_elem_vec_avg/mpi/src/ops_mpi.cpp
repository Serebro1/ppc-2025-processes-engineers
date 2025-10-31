#include "otcheskov_s_elem_vec_avg/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "otcheskov_s_elem_vec_avg/common/include/common.hpp"
#include "util/include/util.hpp"

namespace otcheskov_s_elem_vec_avg {

OtcheskovSElemVecAvgMPI::OtcheskovSElemVecAvgMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = NAN;
}

bool OtcheskovSElemVecAvgMPI::ValidationImpl() {
  return (!GetInput().empty() && std::isnan(GetOutput()));
}

bool OtcheskovSElemVecAvgMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool OtcheskovSElemVecAvgMPI::RunImpl() {
  int ProcRank, ProcNum;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);

  int total_size = GetInput().size();
  int local_size = total_size / ProcNum;
  int remainder = total_size % ProcNum;

  int my_size = local_size + (ProcRank < remainder ? 1 : 0);
  std::vector<int> local_data(my_size);

  std::vector<int> displacements(ProcNum);
  std::vector<int> counts(ProcNum);
  
  int offset = 0;
  for (int i = 0; i < ProcNum; i++) {
    counts[i] = local_size + (i < remainder ? 1 : 0);
    displacements[i] = offset;
    offset += counts[i];
  }

  MPI_Scatterv(GetInput().data(), counts.data(), displacements.data(), MPI_INT,
               local_data.data(), my_size, MPI_INT, 0, MPI_COMM_WORLD);

  int local_sum = std::accumulate(local_data.begin(), local_data.end(), 0);

  int global_sum = 0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  double average = 0;
  if (ProcRank == 0) {
    average = global_sum / static_cast<double>(total_size);
  }

  MPI_Bcast(&average, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = average;

  return true;
}

bool OtcheskovSElemVecAvgMPI::PostProcessingImpl() {
  return true;
}

}  // namespace otcheskov_s_elem_vec_avg
