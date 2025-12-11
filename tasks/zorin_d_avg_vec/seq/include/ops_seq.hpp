#pragma once

#include "task/include/task.hpp"
#include "zorin_d_avg_vec/common/include/common.hpp"

namespace zorin_d_avg_vec {

class ZorinDAvgVecSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ZorinDAvgVecSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zorin_d_avg_vec
