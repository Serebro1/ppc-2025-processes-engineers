#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "task/include/task.hpp"
#include "yakimov_i_max_values_in_matrix_rows/common/include/common.hpp"

namespace yakimov_i_max_values_in_matrix_rows {

class YakimovIMaxValuesInMatrixRowsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit YakimovIMaxValuesInMatrixRowsSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  bool ReadMatrixFromFile(const std::string &filename);

  std::vector<std::vector<InType>> matrix_;
  std::vector<InType> max_Values_;
  size_t rows_{0};
  size_t cols_{0};
  std::string matrix_Filename_;
};

}  // namespace yakimov_i_max_values_in_matrix_rows
