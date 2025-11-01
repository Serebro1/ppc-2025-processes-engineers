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

  int proc_size = local_size + (ProcRank < remainder ? 1 : 0);
  std::vector<int> local_data(proc_size);

  std::vector<int> displacements(ProcNum);
  std::vector<int> counts(ProcNum);

  int offset = 0;
  for (int i = 0; i < ProcNum; i++) {
    counts[i] = local_size + (i < remainder ? 1 : 0);
    displacements[i] = offset;
    offset += counts[i];
  }
  MPI_Scatterv(GetInput().data(), counts.data(), displacements.data(), MPI_INT, local_data.data(), proc_size, MPI_INT,
               0, MPI_COMM_WORLD);
  int local_sum = std::accumulate(local_data.begin(), local_data.end(), 0);

  int total_sum = 0;
  MPI_Allreduce(&local_sum, &total_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = total_sum / static_cast<double>(total_size);
  return true;
}

bool OtcheskovSElemVecAvgMPI::PostProcessingImpl() {
  return true;
}

}  // namespace otcheskov_s_elem_vec_avg
