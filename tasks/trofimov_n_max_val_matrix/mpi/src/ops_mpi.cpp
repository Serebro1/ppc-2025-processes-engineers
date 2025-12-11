#include "trofimov_n_max_val_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "trofimov_n_max_val_matrix/common/include/common.hpp"

namespace trofimov_n_max_val_matrix {

namespace {
constexpr int kRootRank = 0;
}  // namespace

TrofimovNMaxValMatrixMPI::TrofimovNMaxValMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in.begin(), in.end());
  GetOutput() = OutType();
}

bool TrofimovNMaxValMatrixMPI::ValidationImpl() {
  if (GetInput().empty()) {
    return false;
  }

  const std::size_t cols = GetInput()[0].size();
  for (const auto &row : GetInput()) {
    if (row.size() != cols) {
      return false;
    }
  }

  return GetOutput().empty();
}

bool TrofimovNMaxValMatrixMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == kRootRank) {
    GetOutput() = std::vector<int>(GetInput().size(), 0);
  } else {
    GetOutput().clear();
  }

  return true;
}

namespace {

std::tuple<int, int> CalculateLocalRows(int rank, int size, int total_rows) {
  const int rows_per_process = total_rows / size;
  const int remainder = total_rows % size;

  const int start_row = (rank * rows_per_process) + std::min(rank, remainder);
  const int local_rows = rows_per_process + (rank < remainder ? 1 : 0);

  return {start_row, local_rows};
}

std::vector<int> CalculateLocalMaxima(const std::vector<std::vector<int>> &local_input) {
  std::vector<int> local_maxima;
  local_maxima.reserve(local_input.size());

  for (const auto &row : local_input) {
    if (!row.empty()) {
      local_maxima.push_back(*std::ranges::max_element(row));
    }
  }

  return local_maxima;
}

void HandleEmptyCase(int rank, int total_rows) {
  if (rank == kRootRank) {
    std::vector<int> recv_counts(1, 0);
    std::vector<int> displacements(1, 0);
    MPI_Gather(&total_rows, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, kRootRank, MPI_COMM_WORLD);
  } else {
    int zero = 0;
    MPI_Gather(&zero, 1, MPI_INT, nullptr, 0, MPI_INT, kRootRank, MPI_COMM_WORLD);
  }
}

void GatherResults(int rank, int size, int local_rows, const std::vector<int> &local_maxima, std::vector<int> &output,
                   int total_rows) {
  std::vector<int> recv_counts(static_cast<std::size_t>(size), 0);
  std::vector<int> displacements(static_cast<std::size_t>(size), 0);

  MPI_Gather(&local_rows, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, kRootRank, MPI_COMM_WORLD);

  if (rank == kRootRank) {
    displacements[0] = 0;
    for (int i = 1; i < size; ++i) {
      displacements[static_cast<std::size_t>(i)] =
          displacements[static_cast<std::size_t>(i - 1)] + recv_counts[static_cast<std::size_t>(i - 1)];
    }

    if (output.size() != static_cast<std::size_t>(total_rows)) {
      output.resize(static_cast<std::size_t>(total_rows));
    }
  }

  MPI_Gatherv(local_maxima.data(), local_rows, MPI_INT, rank == kRootRank ? output.data() : nullptr, recv_counts.data(),
              displacements.data(), MPI_INT, kRootRank, MPI_COMM_WORLD);
}

bool SendRowsToProcess(int dest, int total_rows, int total_cols, int size,
                       const std::vector<std::vector<int>> &original_input) {
  auto [dest_start_row, dest_local_rows] = CalculateLocalRows(dest, size, total_rows);

  if (dest_local_rows <= 0) {
    return false;
  }

  std::array<int, 2> dest_info = {dest_local_rows, total_cols};
  MPI_Send(dest_info.data(), 2, MPI_INT, dest, 0, MPI_COMM_WORLD);

  for (int i = 0; i < dest_local_rows; ++i) {
    int global_row = dest_start_row + i;
    MPI_Send(original_input[static_cast<std::size_t>(global_row)].data(), total_cols, MPI_INT, dest, i + 1,
             MPI_COMM_WORLD);
  }

  return true;
}

void ProcessRootData(int local_rows, int start_row, const std::vector<std::vector<int>> &original_input,
                     std::vector<std::vector<int>> &local_input) {
  local_input.reserve(static_cast<std::size_t>(local_rows));
  for (int i = 0; i < local_rows; ++i) {
    int global_row = start_row + i;
    local_input.push_back(original_input[static_cast<std::size_t>(global_row)]);
  }
}

bool ReceiveRowsFromRoot(std::vector<std::vector<int>> &local_input) {
  std::array<int, 2> recv_info = {0, 0};
  MPI_Recv(recv_info.data(), 2, MPI_INT, kRootRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  int recv_rows = recv_info[0];
  int recv_cols = recv_info[1];

  if (recv_rows <= 0 || recv_cols <= 0) {
    return false;
  }

  local_input.reserve(static_cast<std::size_t>(recv_rows));
  for (int i = 0; i < recv_rows; ++i) {
    std::vector<int> row(static_cast<std::size_t>(recv_cols));
    MPI_Recv(row.data(), recv_cols, MPI_INT, kRootRank, i + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    local_input.push_back(std::move(row));
  }

  return true;
}

}  // namespace

bool TrofimovNMaxValMatrixMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_rows = 0;
  int total_cols = 0;

  if (rank == kRootRank) {
    const auto &input = GetInput();
    total_rows = static_cast<int>(input.size());
    if (total_rows > 0) {
      total_cols = static_cast<int>(input[0].size());
    }
  }

  MPI_Bcast(&total_rows, 1, MPI_INT, kRootRank, MPI_COMM_WORLD);
  MPI_Bcast(&total_cols, 1, MPI_INT, kRootRank, MPI_COMM_WORLD);

  const bool invalid_matrix = total_rows <= 0 || total_cols <= 0;
  if (invalid_matrix) {
    HandleEmptyCase(rank, total_rows);
    if (rank == kRootRank) {
      GetOutput().clear();
    }
    return true;
  }

  auto [start_row, local_rows] = CalculateLocalRows(rank, size, total_rows);

  if (local_rows <= 0) {
    HandleEmptyCase(rank, total_rows);
    return true;
  }

  std::vector<std::vector<int>> local_input;

  if (rank == kRootRank) {
    const auto &original_input = GetInput();

    for (int dest = 1; dest < size; ++dest) {
      SendRowsToProcess(dest, total_rows, total_cols, size, original_input);
    }

    ProcessRootData(local_rows, start_row, original_input, local_input);
  } else {
    const bool received = ReceiveRowsFromRoot(local_input);
    if (!received) {
      HandleEmptyCase(rank, total_rows);
      return true;
    }
  }

  if (local_input.empty()) {
    HandleEmptyCase(rank, total_rows);
    return true;
  }

  auto local_maxima = CalculateLocalMaxima(local_input);
  GatherResults(rank, size, static_cast<int>(local_input.size()), local_maxima, GetOutput(), total_rows);

  return true;
}

bool TrofimovNMaxValMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace trofimov_n_max_val_matrix
