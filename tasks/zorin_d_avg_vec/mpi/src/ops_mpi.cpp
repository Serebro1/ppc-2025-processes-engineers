#include "zorin_d_avg_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "zorin_d_avg_vec/common/include/common.hpp"

namespace zorin_d_avg_vec {

ZorinDAvgVecMPI::ZorinDAvgVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool ZorinDAvgVecMPI::ValidationImpl() {
  return true;
}

bool ZorinDAvgVecMPI::PreProcessingImpl() {
  return true;
}

bool ZorinDAvgVecMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &vec = GetInput();
  std::size_t total_size = vec.size();

  MPI_Bcast(&total_size, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  std::vector<int> sendcounts(size, 0);
  std::vector<int> displs(size, 0);

  if (rank == 0) {
    std::size_t base = total_size / static_cast<std::size_t>(size);
    std::size_t rem = total_size % static_cast<std::size_t>(size);

    int base_int = static_cast<int>(base);
    int rem_int = static_cast<int>(rem);

    for (int i = 0; i < size; ++i) {
      sendcounts[i] = base_int + (i < rem_int ? 1 : 0);
      displs[i] = (i == 0 ? 0 : displs[i - 1] + sendcounts[i - 1]);
    }
  }

  MPI_Bcast(sendcounts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_vec(sendcounts[rank]);

  MPI_Scatterv(vec.data(), sendcounts.data(), displs.data(), MPI_INT, local_vec.data(), sendcounts[rank], MPI_INT, 0,
               MPI_COMM_WORLD);

  double local_sum = 0.0;
  for (int v : local_vec) {
    local_sum += v;
  }

  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  double avg = 0.0;
  if (total_size != 0) {
    avg = global_sum / static_cast<double>(total_size);
  }

  GetOutput() = avg;

  return true;
}

bool ZorinDAvgVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zorin_d_avg_vec
