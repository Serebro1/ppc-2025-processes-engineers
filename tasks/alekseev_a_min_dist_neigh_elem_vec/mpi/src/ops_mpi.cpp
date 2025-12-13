#include "alekseev_a_min_dist_neigh_elem_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstdlib>
#include <limits>
#include <tuple>
#include <vector>

#include "alekseev_a_min_dist_neigh_elem_vec/common/include/common.hpp"

namespace alekseev_a_min_dist_neigh_elem_vec {

AlekseevAMinDistNeighElemVecMPI::AlekseevAMinDistNeighElemVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(-1, -1);
}

bool AlekseevAMinDistNeighElemVecMPI::ValidationImpl() {
  return true;
}

bool AlekseevAMinDistNeighElemVecMPI::PreProcessingImpl() {
  return true;
}

namespace {
std::vector<int> CalculateSendCountsAndDisplacements(int total_size, int comm_size, std::vector<int> &displacements) {
  int base_chunk = total_size / comm_size;
  int remainder = total_size % comm_size;

  std::vector<int> send_counts(comm_size);
  displacements.resize(comm_size);
  int displacement = 0;
  for (int i = 0; i < comm_size; i++) {
    send_counts[i] = base_chunk + (i < remainder ? 1 : 0);
    displacements[i] = displacement;
    displacement += send_counts[i];
  }
  return send_counts;
}

std::vector<int> CalculatePrevElements(int rank, int comm_size, const std::vector<int> &vec,
                                       const std::vector<int> &displacements) {
  std::vector<int> prev_elements(comm_size, 0);
  if (rank == 0) {
    for (int i = 1; i < comm_size; i++) {
      int prev_block_end = displacements[i] - 1;
      prev_elements[i] = vec[prev_block_end];
    }
  }
  return prev_elements;
}

std::tuple<int, int> FindLocalMinDistance(const std::vector<int> &local_data, int my_chunk_size, int rank,
                                          int my_prev_element, const std::vector<int> &displacements) {
  int local_min_dist = std::numeric_limits<int>::max();
  int local_min_index = -1;

  if (rank > 0 && my_chunk_size > 0) {
    int dist = std::abs(my_prev_element - local_data[0]);
    if (dist < local_min_dist) {
      local_min_dist = dist;
      local_min_index = displacements[rank];
    }
  }

  for (int i = 0; i < my_chunk_size - 1; i++) {
    int dist = std::abs(local_data[i] - local_data[i + 1]);
    if (dist < local_min_dist) {
      local_min_dist = dist;
      local_min_index = displacements[rank] + i + 1;
    }
  }

  return std::make_tuple(local_min_dist, local_min_index);
}
}  // namespace

bool AlekseevAMinDistNeighElemVecMPI::RunImpl() {
  int rank = 0;
  int comm_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  const auto &vec = GetInput();
  int total_size = static_cast<int>(vec.size());

  if (total_size < 2) {
    GetOutput() = std::make_tuple(-1, -1);
    return true;
  }

  std::vector<int> displacements;
  std::vector<int> send_counts = CalculateSendCountsAndDisplacements(total_size, comm_size, displacements);

  int my_chunk_size = send_counts[rank];
  std::vector<int> local_data(my_chunk_size);

  MPI_Scatterv(vec.data(), send_counts.data(), displacements.data(), MPI_INT, local_data.data(), my_chunk_size, MPI_INT,
               0, MPI_COMM_WORLD);

  std::vector<int> prev_elements = CalculatePrevElements(rank, comm_size, vec, displacements);

  int my_prev_element = 0;
  MPI_Scatter(prev_elements.data(), 1, MPI_INT, &my_prev_element, 1, MPI_INT, 0, MPI_COMM_WORLD);

  auto [local_min_dist, local_min_index] =
      FindLocalMinDistance(local_data, my_chunk_size, rank, my_prev_element, displacements);

  int global_min_dist = std::numeric_limits<int>::max();
  MPI_Allreduce(&local_min_dist, &global_min_dist, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  int candidate_index =
      (local_min_dist == global_min_dist && local_min_index != -1) ? local_min_index : std::numeric_limits<int>::max();

  int global_min_index = std::numeric_limits<int>::max();
  MPI_Allreduce(&candidate_index, &global_min_index, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = std::make_tuple(global_min_index - 1, global_min_index);
  return true;
}

bool AlekseevAMinDistNeighElemVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace alekseev_a_min_dist_neigh_elem_vec
