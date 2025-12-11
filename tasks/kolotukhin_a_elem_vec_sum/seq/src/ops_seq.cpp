#include "kolotukhin_a_elem_vec_sum/seq/include/ops_seq.hpp"

#include <cstdint>

#include "kolotukhin_a_elem_vec_sum/common/include/common.hpp"

namespace kolotukhin_a_elem_vec_sum {

KolotukhinAElemVecSumSEQ::KolotukhinAElemVecSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KolotukhinAElemVecSumSEQ::ValidationImpl() {
  return std::equal_to<>()(typeid(GetInput()), typeid(std::uint64_t &));
}

bool KolotukhinAElemVecSumSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool KolotukhinAElemVecSumSEQ::RunImpl() {
  std::uint64_t size = GetInput();
  std::int64_t seed = 42;
  volatile std::int64_t total_sum = 0;
  for (std::uint64_t i = 0; i < size; i++) {
    seed = (seed * 13 + 7) % 10000;
    total_sum += seed;
  }
  GetOutput() = total_sum;
  return true;
}

bool KolotukhinAElemVecSumSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kolotukhin_a_elem_vec_sum
