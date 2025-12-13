#pragma once

#include "kopilov_d_sum_val_col_mat/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kopilov_d_sum_val_col_mat {

class KopilovDSumValColMatSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KopilovDSumValColMatSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kopilov_d_sum_val_col_mat
