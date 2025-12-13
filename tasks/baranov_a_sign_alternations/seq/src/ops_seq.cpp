#include "baranov_a_sign_alternations/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "baranov_a_sign_alternations/common/include/common.hpp"

namespace baranov_a_sign_alternations {

BaranovASignAlternationsSEQ::BaranovASignAlternationsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool BaranovASignAlternationsSEQ::ValidationImpl() {
  return GetOutput() >= 0;
}

bool BaranovASignAlternationsSEQ::PreProcessingImpl() {
  return true;
}

bool BaranovASignAlternationsSEQ::RunImpl() {
  const auto &input = GetInput();

  if (input.size() < 2) {
    GetOutput() = 0;
    return true;
  }

  int alternations_count = 0;

  for (size_t i = 0; i < input.size() - 1; i++) {
    int current = input[i];
    int next = input[i + 1];

    if (current != 0 && next != 0) {
      if ((current > 0 && next < 0) || (current < 0 && next > 0)) {
        alternations_count++;
      }
    }
  }

  GetOutput() = alternations_count;
  return true;
}

bool BaranovASignAlternationsSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace baranov_a_sign_alternations
