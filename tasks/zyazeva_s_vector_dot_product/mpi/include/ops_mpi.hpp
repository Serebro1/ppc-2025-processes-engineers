#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "task/include/task.hpp"
#include "zyazeva_s_vector_dot_product/common/include/common.hpp"

namespace zyazeva_s_vector_dot_product {

class ZyazevaSVecDotProductMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit ZyazevaSVecDotProductMPI(std::vector<std::vector<int>> input)
      : ppc::task::Task<std::vector<std::vector<int>>, int64_t>() {
    SetTypeOfTask(GetStaticTypeOfTask());
    GetInput() = std::move(input);
  }

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zyazeva_s_vector_dot_product
