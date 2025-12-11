#include "kolotukhin_a_elem_vec_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstdint>
#include <numeric>
#include <vector>

#include "kolotukhin_a_elem_vec_sum/common/include/common.hpp"

namespace kolotukhin_a_elem_vec_sum {

KolotukhinAElemVecSumMPI::KolotukhinAElemVecSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KolotukhinAElemVecSumMPI::ValidationImpl() {
  return std::equal_to<>()(typeid(GetInput()), typeid(std::uint64_t &));
}

bool KolotukhinAElemVecSumMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool KolotukhinAElemVecSumMPI::RunImpl() {
  int p_id = -1;
  int pcount = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &p_id);
  MPI_Comm_size(MPI_COMM_WORLD, &pcount);

  std::vector<int> input_data{};
  std::uint64_t input_size = 0;

  if (p_id == 0) {
    input_size = GetInput();
    input_data.resize(input_size);
    int seed = 42;
    for (std::uint64_t i = 0; i < input_size; i++) {
      seed = (seed * 13 + 7) % 10000;
      input_data[i] = seed;
    }
  }

  MPI_Bcast(&input_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  auto p_count = static_cast<uint64_t>(pcount);
  std::uint64_t base_size = input_size / p_count;
  std::uint64_t proc_size = base_size;
  if (p_id == 0) {
    proc_size += input_size % p_count;
  }

  std::vector<int> local_data(proc_size);

  MPI_Scatter(input_data.data(), static_cast<int>(base_size), MPI_INT, local_data.data(), static_cast<int>(base_size),
              MPI_INT, 0, MPI_COMM_WORLD);

  if (p_id == 0 && input_size % p_count != 0) {
    std::uint64_t remainder_start = base_size * p_count;
    std::uint64_t remainder_size = input_size % p_count;
    for (std::uint64_t i = 0; i < remainder_size; i++) {
      local_data[base_size + i] = input_data[remainder_start + i];
    }
  }

  std::int64_t local_sum = std::accumulate(local_data.begin(), local_data.end(), 0LL);
  std::int64_t global_sum = 0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_sum;
  return true;
}

bool KolotukhinAElemVecSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kolotukhin_a_elem_vec_sum
