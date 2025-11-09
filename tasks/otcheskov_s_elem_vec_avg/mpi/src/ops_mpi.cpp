#include "otcheskov_s_elem_vec_avg/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include "otcheskov_s_elem_vec_avg/common/include/common.hpp"

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
  return true;
}

bool OtcheskovSElemVecAvgMPI::RunImpl() {
  if (GetInput().empty()) {
    return false;
  }

  int proc_rank{};
  int proc_num{};
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

  const size_t total_size = GetInput().size();
  const double inv_total = 1.0 / total_size;
  const size_t batch_size = total_size / proc_num;
  const size_t proc_size = batch_size + (proc_rank == proc_num - 1 ? total_size % proc_num : 0);

  auto start_local_data = GetInput().begin() + static_cast<std::vector<int>::difference_type>(proc_rank * batch_size);
  auto end_local_data = start_local_data + static_cast<std::vector<int>::difference_type>(proc_size);

  int local_sum = std::accumulate(start_local_data, end_local_data, 0);
  int total_sum = 0;
  MPI_Request request;
  MPI_Iallreduce(&local_sum, &total_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD, &request);
  MPI_Wait(&request, MPI_STATUS_IGNORE);
  GetOutput() = total_sum * inv_total;
  return !std::isnan(GetOutput());
}

bool OtcheskovSElemVecAvgMPI::PostProcessingImpl() {
  return true;
}

}  // namespace otcheskov_s_elem_vec_avg
