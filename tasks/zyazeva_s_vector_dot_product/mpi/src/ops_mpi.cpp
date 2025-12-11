#include "zyazeva_s_vector_dot_product/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstdint>
#include <vector>

#include "zyazeva_s_vector_dot_product/common/include/common.hpp"

namespace zyazeva_s_vector_dot_product {

namespace {
bool CheckInputValid(const std::vector<std::vector<int32_t>> &input, int64_t &total_elements) {
  if (input.size() < 2) {
    return false;
  }

  const auto &vector1 = input[0];
  const auto &vector2 = input[1];

  if (vector1.size() != vector2.size()) {
    return false;
  }
  if (vector1.empty() || vector2.empty()) {
    return false;
  }

  total_elements = static_cast<int64_t>(vector1.size());
  return true;
}

void CalculateChunkParams(int rank, int size, int64_t total_elements, int64_t &local_size, int64_t &start) {
  const int64_t base_chunk_size = total_elements / size;
  const int64_t remainder = total_elements % size;

  local_size = base_chunk_size + (rank < remainder ? 1 : 0);

  if (rank < remainder) {
    start = rank * (base_chunk_size + 1);
  } else {
    start = (remainder * (base_chunk_size + 1)) + ((rank - remainder) * base_chunk_size);
  }
}

}  // namespace

bool ZyazevaSVecDotProductMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool is_valid = false;

  if (rank == 0) {
    is_valid = true;
  }

  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return is_valid;
}

bool ZyazevaSVecDotProductMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool ZyazevaSVecDotProductMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int64_t total_elements = 0;
  bool error_handling = true;

  if (rank == 0) {
    const auto &input = GetInput();
    error_handling = CheckInputValid(input, total_elements);
    if (!error_handling) {
      GetOutput() = 0;
    }
  }

  int error_flag = error_handling ? 1 : 0;
  MPI_Bcast(&error_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (error_flag == 0) {
    if (rank != 0) {
      GetOutput() = 0;
    }
    return true;
  }

  MPI_Bcast(&total_elements, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);

  int64_t local_size = 0;
  int64_t start = 0;
  CalculateChunkParams(rank, size, total_elements, local_size, start);

  int64_t local_dot_product = 0;
  if (rank == 0) {
    const auto &input = GetInput();
    const auto &vector1_full = input[0];
    const auto &vector2_full = input[1];

    for (int64_t i = start; i < start + local_size; ++i) {
      local_dot_product += static_cast<int64_t>(vector1_full[i]) * static_cast<int64_t>(vector2_full[i]);
    }

    for (int i = 1; i < size; i++) {
      int64_t i_local_size = 0;
      int64_t i_start = 0;
      CalculateChunkParams(i, size, total_elements, i_local_size, i_start);

      MPI_Send(vector1_full.data() + i_start, static_cast<int>(i_local_size), MPI_INT, i, 0, MPI_COMM_WORLD);
      MPI_Send(vector2_full.data() + i_start, static_cast<int>(i_local_size), MPI_INT, i, 1, MPI_COMM_WORLD);
    }
  } else {
    std::vector<int32_t> local_vector1(local_size);
    std::vector<int32_t> local_vector2(local_size);

    MPI_Recv(local_vector1.data(), static_cast<int>(local_size), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(local_vector2.data(), static_cast<int>(local_size), MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int64_t i = 0; i < local_size; ++i) {
      local_dot_product += static_cast<int64_t>(local_vector1[i]) * static_cast<int64_t>(local_vector2[i]);
    }
  }

  int64_t global_dot_product = 0;
  MPI_Allreduce(&local_dot_product, &global_dot_product, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = static_cast<OutType>(global_dot_product);
  return true;
}

bool ZyazevaSVecDotProductMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zyazeva_s_vector_dot_product
