#include "iskhakov_d_trapezoidal_integration/seq/include/ops_seq.hpp"

#include <cmath>

#include "iskhakov_d_trapezoidal_integration/common/include/common.hpp"

namespace iskhakov_d_trapezoidal_integration {

IskhakovDTrapezoidalIntegrationSEQ::IskhakovDTrapezoidalIntegrationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool IskhakovDTrapezoidalIntegrationSEQ::ValidationImpl() {
  auto &input = GetInput();

  double lower_level = std::get<0>(input);
  double top_level = std::get<1>(input);
  int number_steps = std::get<3>(input);

  return (lower_level < top_level) && (number_steps > 0);
}

bool IskhakovDTrapezoidalIntegrationSEQ::PreProcessingImpl() {
  return GetOutput() == 0.0;
}

bool IskhakovDTrapezoidalIntegrationSEQ::RunImpl() {
  auto &input = GetInput();

  double lower_level = std::get<0>(input);
  double top_level = std::get<1>(input);
  auto input_function = std::get<2>(input);
  int number_steps = std::get<3>(input);

  double result = 0.0;
  double step = (top_level - lower_level) / static_cast<double>(number_steps);

  result = (input_function(lower_level) + input_function(top_level)) / 2.0;

  for (int step_index = 1; step_index < number_steps; ++step_index) {
    result += input_function(lower_level + (step * step_index));
  }

  result *= step;
  GetOutput() = result;

  return true;
}

bool IskhakovDTrapezoidalIntegrationSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace iskhakov_d_trapezoidal_integration
