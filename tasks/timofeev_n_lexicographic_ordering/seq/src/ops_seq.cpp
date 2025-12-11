#include "timofeev_n_lexicographic_ordering/seq/include/ops_seq.hpp"

#include <cstddef>
#include <utility>

#include "timofeev_n_lexicographic_ordering/common/include/common.hpp"

namespace timofeev_n_lexicographic_ordering {

TimofeevNLexicographicOrderingSEQ::TimofeevNLexicographicOrderingSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::pair<int, int>(1, 1);
}

bool TimofeevNLexicographicOrderingSEQ::ValidationImpl() {
  return true;
}

bool TimofeevNLexicographicOrderingSEQ::PreProcessingImpl() {
  return true;
}

bool TimofeevNLexicographicOrderingSEQ::RunImpl() {
  auto input = GetInput();

  // only true if comparison is true on every step
  for (size_t i = 0; !input.first.empty() && i < input.first.length() - 1; i++) {
    GetOutput().first &= static_cast<int>(input.first[i] <= input.first[i + 1]);
  }
  for (size_t i = 0; !input.second.empty() && i < input.second.length() - 1; i++) {
    GetOutput().second &= static_cast<int>(input.second[i] <= input.second[i + 1]);
  }

  return true;
}

bool TimofeevNLexicographicOrderingSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace timofeev_n_lexicographic_ordering
