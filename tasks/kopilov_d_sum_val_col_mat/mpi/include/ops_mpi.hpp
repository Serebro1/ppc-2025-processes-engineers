#pragma once

#include "kopilov_d_sum_val_col_mat/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kopilov_d_sum_val_col_mat {

class KopilovDSumValColMatMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KopilovDSumValColMatMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kopilov_d_sum_val_col_mat
