#include "shemetov_d_find_error_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <vector>

#include "shemetov_d_find_error_vec/common/include/common.hpp"

namespace shemetov_d_find_error_vec {

namespace {

constexpr double kEpsilon = 1e-10;

}  // namespace

int ShemetovDFindErrorVecMPI::DetectDrop(double left, double right) noexcept {
  return (left > right + kEpsilon) ? 1 : 0;
}

ShemetovDFindErrorVecMPI::ShemetovDFindErrorVecMPI(const InType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
  GetOutput() = 0;
}

bool ShemetovDFindErrorVecMPI::ValidationImpl() {
  return true;
}

bool ShemetovDFindErrorVecMPI::PreProcessingImpl() {
  return true;
}

bool ShemetovDFindErrorVecMPI::RunImpl() {
  const auto &data = GetInput();
  const int data_size = static_cast<int>(data.size());

  if (data_size < 2) {
    GetOutput() = 0;
    return true;
  }

  int world_rank = 0;
  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::vector<int> sendcounts(world_size);
  std::vector<int> displs(world_size);

  int base = data_size / world_size;
  int extra = data_size % world_size;

  for (int rank_idx = 0; rank_idx < world_size; rank_idx++) {
    sendcounts[rank_idx] = base + (rank_idx < extra ? 1 : 0);
    displs[rank_idx] = (rank_idx * base) + std::min(rank_idx, extra);
  }

  int local_size = sendcounts[world_rank];
  std::vector<double> local_data(local_size);

  MPI_Scatterv(data.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, local_data.data(), local_size, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  int local_viol = 0;

  for (int i = 0; i + 1 < local_size; i++) {
    local_viol += DetectDrop(local_data[i], local_data[i + 1]);
  }

  if (world_rank > 0 && local_size > 0 && displs[world_rank] > 0) {
    double left = data[displs[world_rank] - 1];
    double right = local_data[0];
    local_viol += DetectDrop(left, right);
  }

  int global_viol = 0;
  MPI_Allreduce(&local_viol, &global_viol, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_viol;
  return true;
}

bool ShemetovDFindErrorVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shemetov_d_find_error_vec
