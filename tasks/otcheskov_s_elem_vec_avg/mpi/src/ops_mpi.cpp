#include "otcheskov_s_elem_vec_avg/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "otcheskov_s_elem_vec_avg/common/include/common.hpp"

namespace otcheskov_s_elem_vec_avg {

OtcheskovSElemVecAvgMPI::OtcheskovSElemVecAvgMPI(const InType &in) {
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num_);
  SetTypeOfTask(GetStaticTypeOfTask());
  if (proc_rank_ == 0) {
    GetInput() = in;
  }
  GetOutput() = NAN;
}

bool OtcheskovSElemVecAvgMPI::ValidationImpl() {
  bool is_valid = true;
  if (proc_rank_ == 0) {
    is_valid = !GetInput().empty() && std::isnan(GetOutput());
  }
  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return is_valid;
}

bool OtcheskovSElemVecAvgMPI::PreProcessingImpl() {
  return true;
}

bool OtcheskovSElemVecAvgMPI::RunImpl() {
  int total_size = 0;
  if (proc_rank_ == 0) {
    total_size = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  ComputeDistribution(total_size);

  int64_t local_sum{};
  for (int val : local_data_) {
    local_sum += val;
  }

  int64_t total_sum{};
  MPI_Allreduce(&local_sum, &total_sum, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = static_cast<double>(total_sum) / static_cast<double>(total_size);

  return !std::isnan(GetOutput());
}

bool OtcheskovSElemVecAvgMPI::PostProcessingImpl() {
  return true;
}

void OtcheskovSElemVecAvgMPI::ComputeDistribution(int total_size) {
  int local_size = total_size / proc_num_;
  int remainder = total_size % proc_num_;
  int proc_size = local_size + (proc_rank_ < remainder ? 1 : 0);

  local_data_.resize(proc_size);
  counts_.resize(proc_num_);
  displacements_.resize(proc_num_);
  if (proc_rank_ == 0) {
    int offset = 0;
    for (int i = 0; i < proc_num_; i++) {
      counts_[i] = local_size + (i < remainder ? 1 : 0);
      displacements_[i] = offset;
      offset += counts_[i];
    }
  }
  MPI_Bcast(counts_.data(), proc_num_, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displacements_.data(), proc_num_, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(GetInput().data(), counts_.data(), displacements_.data(), MPI_INT, local_data_.data(), proc_size,
               MPI_INT, 0, MPI_COMM_WORLD);
}

}  // namespace otcheskov_s_elem_vec_avg
