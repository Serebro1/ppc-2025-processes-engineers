#pragma once

#include <cstdint>
#include <vector>

#include "task/include/task.hpp"
#include "zyazeva_s_vector_dot_product/common/include/common.hpp"

namespace zyazeva_s_vector_dot_product {

class ZyazevaSVecDotProductSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ZyazevaSVecDotProductSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  std::vector<std::vector<int64_t>> input_;
};

}  // namespace zyazeva_s_vector_dot_product
