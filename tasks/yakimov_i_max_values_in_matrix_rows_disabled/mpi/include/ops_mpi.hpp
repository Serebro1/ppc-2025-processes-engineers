#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "task/include/task.hpp"
#include "yakimov_i_max_values_in_matrix_rows/common/include/common.hpp"

namespace yakimov_i_max_values_in_matrix_rows {

class YakimovIMaxValuesInMatrixRowsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit YakimovIMaxValuesInMatrixRowsMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  bool ReadMatrixFromFile(const std::string &filename);
  // вспомогательные функции к RunImpl
  void BroadcastMatrixDimensions(int &total_rows, int &total_cols) const;
  void SendDataToWorkers(int size, int total_rows, int total_cols);
  void ProcessLocalData(int rank, int local_rows, int total_cols, int start_row, std::vector<InType> &local_data);
  void GatherResults(int rank, int size, int total_rows, const std::vector<InType> &local_max_values, int start_row,
                     int local_rows);

  std::vector<std::vector<InType>> matrix_;
  std::vector<InType> max_Values_;
  size_t rows_{0};
  size_t cols_{0};
  std::string matrix_Filename_;
};

}  // namespace yakimov_i_max_values_in_matrix_rows
