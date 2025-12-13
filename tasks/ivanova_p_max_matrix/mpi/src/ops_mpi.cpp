#include "ivanova_p_max_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstring>  // memcpy
#include <limits>
#include <vector>

#include "ivanova_p_max_matrix/common/include/common.hpp"

namespace ivanova_p_max_matrix {

IvanovaPMaxMatrixMPI::IvanovaPMaxMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetOutput() = std::numeric_limits<int>::min();

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetInput() = in;  // Только root хранит входную матрицу
  } else {
    GetInput().clear();  // Остальные — пустая матрица
  }
}

bool IvanovaPMaxMatrixMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool ok = true;

  if (rank == 0) {
    if (GetInput().empty()) {
      ok = false;
    } else {
      const size_t cols = GetInput()[0].size();
      for (const auto &row : GetInput()) {
        if (row.size() != cols) {
          ok = false;
          break;
        }
      }
    }
  }

  int ok_int = ok ? 1 : 0;
  MPI_Bcast(&ok_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return ok_int != 0;
}

bool IvanovaPMaxMatrixMPI::PreProcessingImpl() {
  GetOutput() = std::numeric_limits<int>::min();
  return true;
}

namespace {

std::vector<int> PackMatrix(const InType &input, int rows, int cols) {
  std::vector<int> flat(static_cast<size_t>(rows) * cols);

  size_t pos = 0;
  for (int i = 0; i < rows; ++i) {
    // быстрее, чем двойной for: memcpy целой строки
    std::memcpy(&flat[pos], input[i].data(), sizeof(int) * cols);
    pos += cols;
  }
  return flat;
}

int FindLocalMax(const std::vector<int> &vec) {
  if (vec.empty()) {
    return std::numeric_limits<int>::min();
  }
  return *std::ranges::max_element(vec);
}

}  // namespace

bool IvanovaPMaxMatrixMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows = 0;
  int cols = 0;
  if (rank == 0) {
    rows = static_cast<int>(GetInput().size());
    cols = rows > 0 ? static_cast<int>(GetInput()[0].size()) : 0;
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int total = rows * cols;

  int base = total / size;
  int rem = total % size;

  int my_count = base + (rank < rem ? 1 : 0);

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);

  for (int rank_idx = 0; rank_idx < size; rank_idx++) {
    sendcounts[rank_idx] = base + (rank_idx < rem ? 1 : 0);
  }

  displs[0] = 0;
  for (int rank_idx = 1; rank_idx < size; rank_idx++) {
    displs[rank_idx] = displs[rank_idx - 1] + sendcounts[rank_idx - 1];
  }

  std::vector<int> flat;
  if (rank == 0) {
    flat = PackMatrix(GetInput(), rows, cols);
  }

  std::vector<int> local(my_count);

  MPI_Scatterv(rank == 0 ? flat.data() : nullptr, sendcounts.data(), displs.data(), MPI_INT, local.data(), my_count,
               MPI_INT, 0, MPI_COMM_WORLD);

  int local_max = FindLocalMax(local);

  int global_max = 0;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  GetOutput() = global_max;
  return true;
}

bool IvanovaPMaxMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace ivanova_p_max_matrix
