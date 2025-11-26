#include "viderman_a_elem_vec_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace viderman_a_elem_vec_sum {

VidermanAElemVecSumMPI::VidermanAElemVecSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool VidermanAElemVecSumMPI::ValidationImpl() {
  return (GetOutput() == 0.0);
}

bool VidermanAElemVecSumMPI::PreProcessingImpl() {
  return true;
}

bool VidermanAElemVecSumMPI::RunImpl() {
  const auto &input_vector = GetInput();
  if (input_vector.empty()) {
    GetOutput() = 0.0;
    return true;
  }

  int my_rank = 0, total_processes = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &total_processes);

  const size_t element_count = input_vector.size();
  const auto total_procs_size = static_cast<size_t>(total_processes);

  // процессов больше чем элементов
  if (total_procs_size > element_count) {
    if (static_cast<size_t>(my_rank) < element_count) {
      double single_element = input_vector[static_cast<size_t>(my_rank)];
      double final_result = 0.0;
      MPI_Allreduce(&single_element, &final_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      GetOutput() = final_result;
    } else {
      double zero = 0.0;
      double final_result = 0.0;
      MPI_Allreduce(&zero, &final_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      GetOutput() = final_result;
    }
    return true;
  }

  const size_t base_chunk = element_count / total_procs_size;
  const size_t remaining_elements = element_count % total_procs_size;
  size_t my_chunk_size = base_chunk;
  if (static_cast<size_t>(my_rank) < remaining_elements) {
    my_chunk_size = base_chunk + 1;
  }

  size_t start_position = static_cast<size_t>(my_rank) * base_chunk;
  if (static_cast<size_t>(my_rank) <= remaining_elements && remaining_elements > 0) {
    start_position += static_cast<size_t>(my_rank);
  } else {
    start_position += remaining_elements;
  }

  double process_sum = 0.0;
  auto segment_start = input_vector.begin() + start_position;
  auto segment_end = segment_start + my_chunk_size;

  for (auto it = segment_start; it != segment_end; ++it) {
    process_sum += *it;
  }

  double final_result = 0.0;
  MPI_Allreduce(&process_sum, &final_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = final_result;
  return true;
}
bool VidermanAElemVecSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace viderman_a_elem_vec_sum
