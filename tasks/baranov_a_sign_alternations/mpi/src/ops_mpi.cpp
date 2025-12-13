#include "baranov_a_sign_alternations/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

#include "baranov_a_sign_alternations/common/include/common.hpp"

namespace baranov_a_sign_alternations {

namespace {
int CountAlternationsInRange(const std::vector<int> &input, int start, int end) {
  int count = 0;
  for (int i = start; i < end && i < static_cast<int>(input.size()) - 1; i++) {
    int current = input[i];
    int next = input[i + 1];
    if (current != 0 && next != 0) {
      if ((current > 0 && next < 0) || (current < 0 && next > 0)) {
        count++;
      }
    }
  }
  return count;
}

void CalculateChunkBounds(int world_rank, int world_size, int pairs_count, int &start_pair, int &end_pair) {
  int pairs_per_process = pairs_count / world_size;
  int remainder = pairs_count % world_size;

  if (world_rank < remainder) {
    start_pair = world_rank * (pairs_per_process + 1);
    end_pair = start_pair + pairs_per_process + 1;
  } else {
    start_pair = (remainder * (pairs_per_process + 1)) + ((world_rank - remainder) * pairs_per_process);
    end_pair = start_pair + pairs_per_process;
  }
}
}  // namespace

BaranovASignAlternationsMPI::BaranovASignAlternationsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool BaranovASignAlternationsMPI::ValidationImpl() {
  return GetOutput() >= 0;
}

bool BaranovASignAlternationsMPI::PreProcessingImpl() {
  return true;
}

bool BaranovASignAlternationsMPI::RunImpl() {
  const auto &input = GetInput();

  int world_size = 0;
  int world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (input.size() < 2) {
    GetOutput() = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  int pairs_count = static_cast<int>(input.size()) - 1;

  if (pairs_count < world_size) {
    int alternations_count = 0;
    if (world_rank == 0) {
      alternations_count = CountAlternationsInRange(input, 0, pairs_count);
    }

    MPI_Bcast(&alternations_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    GetOutput() = alternations_count;
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  int start_pair = 0;
  int end_pair = 0;
  CalculateChunkBounds(world_rank, world_size, pairs_count, start_pair, end_pair);

  int local_alternations = CountAlternationsInRange(input, start_pair, end_pair);

  int total_alternations = 0;

  MPI_Reduce(&local_alternations, &total_alternations, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    GetOutput() = total_alternations;
    for (int i = 1; i < world_size; i++) {
      MPI_Send(&total_alternations, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }

  } else {
    MPI_Recv(&total_alternations, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    GetOutput() = total_alternations;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool BaranovASignAlternationsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace baranov_a_sign_alternations
