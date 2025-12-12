#include "tsyplakov_k_vec_neighbours/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "tsyplakov_k_vec_neighbours/common/include/common.hpp"

namespace tsyplakov_k_vec_neighbours {

TsyplakovKVecNeighboursSEQ::TsyplakovKVecNeighboursSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(-1, -1);
}

bool TsyplakovKVecNeighboursSEQ::ValidationImpl() {
  return true;
}

bool TsyplakovKVecNeighboursSEQ::PreProcessingImpl() {
  vector_data_ = GetInput();
  return true;
}

bool TsyplakovKVecNeighboursSEQ::RunImpl() {
  const std::size_t n = vector_data_.size();

  if (n < 2) {
    GetOutput() = std::make_tuple(-1, -1);
    return true;
  }

  int min_diff = std::numeric_limits<int>::max();
  int min_index = -1;

  for (std::size_t i = 0; i + 1 < n; ++i) {
    int64_t diff = std::llabs(static_cast<int64_t>(vector_data_[i + 1]) - static_cast<int64_t>(vector_data_[i]));

    if (diff < min_diff || (diff == min_diff && std::cmp_less(i, static_cast<std::size_t>(min_index)))) {
      min_diff = static_cast<int>(diff);
      min_index = static_cast<int>(i);
    }
  }

  if (min_index >= 0) {
    GetOutput() = std::make_tuple(min_index, min_index + 1);
  } else {
    GetOutput() = std::make_tuple(-1, -1);
  }

  return true;
}

bool TsyplakovKVecNeighboursSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace tsyplakov_k_vec_neighbours
