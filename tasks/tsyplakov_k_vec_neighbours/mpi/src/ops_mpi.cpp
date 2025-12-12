#include "tsyplakov_k_vec_neighbours/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstdlib>
#include <limits>
#include <tuple>
#include <vector>

#include "tsyplakov_k_vec_neighbours/common/include/common.hpp"

namespace tsyplakov_k_vec_neighbours {

TsyplakovKVecNeighboursMPI::TsyplakovKVecNeighboursMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(-1, -1);
}

bool TsyplakovKVecNeighboursMPI::ValidationImpl() {
  return true;
}

bool TsyplakovKVecNeighboursMPI::PreProcessingImpl() {
  return true;
}

bool TsyplakovKVecNeighboursMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &full_arr = GetInput();
  int n = static_cast<int>(full_arr.size());

  if (n < 2) {
    GetOutput() = std::make_tuple(-1, -1);
    return true;
  }

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);

  int base = n / size;
  int extra = n % size;

  int offset = 0;
  for (int i = 0; i < size; i++) {
    sendcounts[i] = base + (i < extra ? 1 : 0);
    displs[i] = offset;
    offset += sendcounts[i];
  }

  for (int i = 0; i < size - 1; i++) {
    sendcounts[i]++;
  }

  int recv_count = sendcounts[rank];
  std::vector<int> local_arr(recv_count);

  MPI_Scatterv(full_arr.data(), sendcounts.data(), displs.data(), MPI_INT, local_arr.data(), recv_count, MPI_INT, 0,
               MPI_COMM_WORLD);

  int best_local_gap = std::numeric_limits<int>::max();
  int best_local_pos = -1;

  int global_offset = displs[rank];

  for (int i = 0; i < recv_count - 1; i++) {
    int diff = std::abs(local_arr[i] - local_arr[i + 1]);
    if (diff < best_local_gap) {
      best_local_gap = diff;
      best_local_pos = global_offset + i;
    }
  }

  int best_global_gap = 0;
  MPI_Allreduce(&best_local_gap, &best_global_gap, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  int candidate_idx = (best_local_gap == best_global_gap ? best_local_pos : std::numeric_limits<int>::max());
  int final_idx = 0;

  MPI_Allreduce(&candidate_idx, &final_idx, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = std::make_tuple(final_idx, final_idx + 1);
  return true;
}

bool TsyplakovKVecNeighboursMPI::PostProcessingImpl() {
  return true;
}

}  // namespace tsyplakov_k_vec_neighbours
