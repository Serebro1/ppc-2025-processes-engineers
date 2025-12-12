#include "iskhakov_d_trapezoidal_integration/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "iskhakov_d_trapezoidal_integration/common/include/common.hpp"

namespace iskhakov_d_trapezoidal_integration {

IskhakovDTrapezoidalIntegrationMPI::IskhakovDTrapezoidalIntegrationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool IskhakovDTrapezoidalIntegrationMPI::ValidationImpl() {
  auto &input = GetInput();
  double lower_level = std::get<0>(input);
  double top_level = std::get<1>(input);
  int number_steps = std::get<3>(input);

  return (lower_level < top_level) && (number_steps > 0);
}

bool IskhakovDTrapezoidalIntegrationMPI::PreProcessingImpl() {
  return true;
}

bool IskhakovDTrapezoidalIntegrationMPI::RunImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  auto &input = GetInput();

  double lower_level = 0.0;
  double top_level = 0.0;
  int number_steps = 0;

  if (world_rank == 0) {
    lower_level = std::get<0>(input);
    top_level = std::get<1>(input);
    number_steps = std::get<3>(input);
  }

  MPI_Bcast(&lower_level, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&top_level, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&number_steps, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (number_steps == 0) {
    GetOutput() = 0.0;
    return true;
  }

  auto input_function = std::get<2>(input);
  double step = (top_level - lower_level) / static_cast<double>(number_steps);

  int base_count = number_steps / world_size;
  int remainder = number_steps % world_size;

  std::vector<int> elements_per_proc(static_cast<std::size_t>(world_size));
  std::vector<int> displacement(static_cast<std::size_t>(world_size));

  int offset = 0;
  for (int process_id = 0; process_id < world_size; ++process_id) {
    elements_per_proc[static_cast<std::size_t>(process_id)] = base_count + (process_id < remainder ? 1 : 0);
    displacement[static_cast<std::size_t>(process_id)] = offset;
    offset += elements_per_proc[static_cast<std::size_t>(process_id)];
  }

  int local_count = elements_per_proc[static_cast<std::size_t>(world_rank)];

  std::vector<double> all_points;
  if (world_rank == 0) {
    all_points.resize(static_cast<std::size_t>(number_steps) + 1);
    for (int step_index = 0; step_index <= number_steps; ++step_index) {
      all_points[static_cast<std::size_t>(step_index)] = lower_level + (static_cast<double>(step_index) * step);
    }
  }

  std::vector<double> local_points(static_cast<std::size_t>(local_count));
  MPI_Scatterv(all_points.data(), elements_per_proc.data(), displacement.data(), MPI_DOUBLE, local_points.data(),
               local_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double local_sum = 0.0;
  for (double point : local_points) {
    local_sum += input_function(point);
  }

  if (world_rank == 0) {
    local_sum -= input_function(local_points.front()) * 0.5;
  }
  if (world_rank == world_size - 1) {
    local_sum -= input_function(local_points.back()) * 0.5;
  }

  double result = 0.0;
  MPI_Allreduce(&local_sum, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = result * step;

  return true;
}

bool IskhakovDTrapezoidalIntegrationMPI::PostProcessingImpl() {
  return true;
}

}  // namespace iskhakov_d_trapezoidal_integration
