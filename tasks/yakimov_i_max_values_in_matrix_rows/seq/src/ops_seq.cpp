#include "yakimov_i_max_values_in_matrix_rows/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

// #include "util/include/util.hpp"
#include "yakimov_i_max_values_in_matrix_rows/common/include/common.hpp"

namespace yakimov_i_max_values_in_matrix_rows {

YakimovIMaxValuesInMatrixRowsSEQ::YakimovIMaxValuesInMatrixRowsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
  std::filesystem::path base_path = std::filesystem::current_path();
  while (base_path.filename() != "ppc-2025-processes-engineers") {
    base_path = base_path.parent_path();
  }
  matrix_Filename_ =
      base_path.string() + "/tasks/yakimov_i_max_values_in_matrix_rows/data/" + std::to_string(GetInput()) + ".txt";
}

bool YakimovIMaxValuesInMatrixRowsSEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool YakimovIMaxValuesInMatrixRowsSEQ::PreProcessingImpl() {
  if (!ReadMatrixFromFile(matrix_Filename_)) {
    return false;
  }

  matrix_.resize(rows_);
  max_Values_.resize(rows_);
  return true;
}

bool YakimovIMaxValuesInMatrixRowsSEQ::ReadMatrixFromFile(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    return false;
  }

  file >> rows_ >> cols_;

  if (rows_ == 0 || cols_ == 0) {
    return false;
  }

  matrix_.resize(rows_);
  for (size_t i = 0; i < rows_; i++) {
    matrix_[i].resize(cols_);
  }

  for (size_t i = 0; i < rows_; i++) {
    for (size_t j = 0; j < cols_; j++) {
      if (!(file >> matrix_[i][j])) {
        return false;
      }
    }
  }

  file.close();
  return true;
}

bool YakimovIMaxValuesInMatrixRowsSEQ::RunImpl() {
  for (size_t i = 0; i < rows_; i++) {
    max_Values_[i] = matrix_[i][0];
    for (size_t j = 1; j < cols_; j++) {
      max_Values_[i] = std::max(matrix_[i][j], max_Values_[i]);
    }
  }
  return true;
}

bool YakimovIMaxValuesInMatrixRowsSEQ::PostProcessingImpl() {
  if (!max_Values_.empty()) {
    OutType result = 0;
    for (const auto &max_val : max_Values_) {
      result += max_val;
    }
    GetOutput() = result;
    return true;
  }
  return false;
}

}  // namespace yakimov_i_max_values_in_matrix_rows
