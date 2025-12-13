#include "shemetov_d_find_error_vec/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "shemetov_d_find_error_vec/common/include/common.hpp"

namespace shemetov_d_find_error_vec {

namespace {
constexpr double kEpsilon = 1e-10;
}  // namespace

ShemetovDFindErrorVecSEQ::ShemetovDFindErrorVecSEQ(const InType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
  GetOutput() = 0;
}

bool ShemetovDFindErrorVecSEQ::ValidationImpl() {
  return true;
}

bool ShemetovDFindErrorVecSEQ::PreProcessingImpl() {
  return true;
}

bool ShemetovDFindErrorVecSEQ::RunImpl() {
  const auto &data = GetInput();
  const size_t size = data.size();

  if (size < 2) {
    GetOutput() = 0;
    return true;
  }

  int violations = 0;

  for (size_t i = 0; i + 1 < size; i += 1) {
    if (data[i] > data[i + 1] + kEpsilon) {
      violations += 1;
    }
  }

  GetOutput() = violations;
  return true;
}

bool ShemetovDFindErrorVecSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace shemetov_d_find_error_vec
