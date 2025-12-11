#include "zyazeva_s_vector_dot_product/seq/include/ops_seq.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "zyazeva_s_vector_dot_product/common/include/common.hpp"

namespace zyazeva_s_vector_dot_product {

namespace {
bool CheckInputValid(const std::vector<std::vector<int32_t>> &input) {
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

  return true;
}
}  // namespace

ZyazevaSVecDotProductSEQ::ZyazevaSVecDotProductSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType temp = in;
  GetInput() = std::move(temp);
  GetOutput() = 0;
}

bool ZyazevaSVecDotProductSEQ::ValidationImpl() {
  return true;
}

bool ZyazevaSVecDotProductSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool ZyazevaSVecDotProductSEQ::RunImpl() {
  auto &input = GetInput();
  auto &vec1 = input[0];
  auto &vec2 = input[1];
  if (!CheckInputValid(input)) {
    GetOutput() = 0;
    return true;
  }

  int64_t dot_product = 0;

  for (size_t i = 0; i < vec1.size(); i++) {
    dot_product += static_cast<int64_t>(vec1[i]) * static_cast<int64_t>(vec2[i]);
  }

  GetOutput() = dot_product;
  return true;
}

bool ZyazevaSVecDotProductSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zyazeva_s_vector_dot_product
