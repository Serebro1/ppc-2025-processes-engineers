#include "yakimov_i_max_values_in_matrix_rows/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// #include "util/include/util.hpp"
#include "yakimov_i_max_values_in_matrix_rows/common/include/common.hpp"

namespace yakimov_i_max_values_in_matrix_rows {

namespace {
void CalculateLocalRows(int rank, int size, int total_rows, int &local_rows, int &start_row, int &end_row) {
  int rows_per_process = total_rows / size;
  int remainder = total_rows % size;

  start_row = (rank * rows_per_process) + std::min(rank, remainder);
  end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);
  local_rows = end_row - start_row;
}

void FindLocalMaxValues(int local_rows, int total_cols, const std::vector<InType> &local_data,
                        std::vector<InType> &local_max_values) {
  for (int i = 0; i < local_rows; i++) {
    InType row_max = local_data[static_cast<size_t>(i) * static_cast<size_t>(total_cols)];
    for (int j = 1; j < total_cols; j++) {
      row_max = std::max(local_data[(i * total_cols) + j], row_max);
    }
    local_max_values[i] = row_max;
  }
}
}  // namespace

bool YakimovIMaxValuesInMatrixRowsMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return (GetInput() > 0);
  }
  return true;
}

void YakimovIMaxValuesInMatrixRowsMPI::SendDataToWorkers(int size, int total_rows, int total_cols) {
  int rows_per_process = total_rows / size;
  int remainder = total_rows % size;

  for (int proc = 1; proc < size; proc++) {
    int proc_start = (proc * rows_per_process) + std::min(proc, remainder);
    int proc_rows = rows_per_process + (proc < remainder ? 1 : 0);

    std::vector<InType> proc_data;
    proc_data.reserve(static_cast<size_t>(proc_rows) * static_cast<size_t>(total_cols));
    for (int i = 0; i < proc_rows; i++) {
      proc_data.insert(proc_data.end(), matrix_[proc_start + i].begin(), matrix_[proc_start + i].end());
    }

    MPI_Send(proc_data.data(), proc_rows * total_cols, MPI_INT, proc, 0, MPI_COMM_WORLD);
  }
}

void YakimovIMaxValuesInMatrixRowsMPI::ProcessLocalData(int rank, int local_rows, int total_cols, int start_row,
                                                        std::vector<InType> &local_data) {
  if (rank == 0) {
    local_data.reserve(static_cast<size_t>(local_rows) * static_cast<size_t>(total_cols));
    for (int i = 0; i < local_rows; i++) {
      local_data.insert(local_data.end(), matrix_[start_row + i].begin(), matrix_[start_row + i].end());
    }
  } else {
    local_data.resize(static_cast<size_t>(local_rows) * static_cast<size_t>(total_cols));
    MPI_Recv(local_data.data(), local_rows * total_cols, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void YakimovIMaxValuesInMatrixRowsMPI::GatherResults(int rank, int size, int total_rows,
                                                     const std::vector<InType> &local_max_values, int start_row,
                                                     int local_rows) {
  int rows_per_process = total_rows / size;
  int remainder = total_rows % size;

  if (rank == 0) {
    max_Values_.resize(total_rows);

    for (int i = 0; i < local_rows; i++) {
      max_Values_[start_row + i] = local_max_values[i];
    }

    for (int proc = 1; proc < size; proc++) {
      int proc_start = (proc * rows_per_process) + std::min(proc, remainder);
      int proc_rows = rows_per_process + (proc < remainder ? 1 : 0);

      std::vector<InType> proc_max_values(proc_rows);
      MPI_Recv(proc_max_values.data(), proc_rows, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int i = 0; i < proc_rows; i++) {
        max_Values_[proc_start + i] = proc_max_values[i];
      }
    }
  } else {
    MPI_Send(local_max_values.data(), local_rows, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
}

bool YakimovIMaxValuesInMatrixRowsMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    if (!max_Values_.empty()) {
      OutType result = 0;
      for (const auto &max_val : max_Values_) {
        result += max_val;
      }
      GetOutput() = result;
    } else {
      return false;
    }
  } else {
    GetOutput() = 0;
  }

  OutType final_result = GetOutput();
  MPI_Bcast(&final_result, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = final_result;

  return true;
}

YakimovIMaxValuesInMatrixRowsMPI::YakimovIMaxValuesInMatrixRowsMPI(const InType &in) {
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

bool YakimovIMaxValuesInMatrixRowsMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    if (!ReadMatrixFromFile(matrix_Filename_)) {
      return false;
    }

    max_Values_.resize(rows_, 0);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool YakimovIMaxValuesInMatrixRowsMPI::ReadMatrixFromFile(const std::string &filename) {
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

void YakimovIMaxValuesInMatrixRowsMPI::BroadcastMatrixDimensions(int &total_rows, int &total_cols) const {
  std::array<int, 2> matrix_dims = {0, 0};
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    matrix_dims[0] = static_cast<int>(rows_);
    matrix_dims[1] = static_cast<int>(cols_);
  }

  MPI_Bcast(matrix_dims.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
  total_rows = matrix_dims[0];
  total_cols = matrix_dims[1];
}

bool YakimovIMaxValuesInMatrixRowsMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_rows = 0;
  int total_cols = 0;
  BroadcastMatrixDimensions(total_rows, total_cols);

  if (total_rows == 0 || total_cols == 0) {
    return false;
  }

  int local_rows = 0;
  int start_row = 0;
  int end_row = 0;
  CalculateLocalRows(rank, size, total_rows, local_rows, start_row, end_row);

  if (rank == 0) {
    SendDataToWorkers(size, total_rows, total_cols);
  }

  std::vector<InType> local_data;
  ProcessLocalData(rank, local_rows, total_cols, start_row, local_data);

  std::vector<InType> local_max_values(local_rows);
  FindLocalMaxValues(local_rows, total_cols, local_data, local_max_values);

  GatherResults(rank, size, total_rows, local_max_values, start_row, local_rows);

  return true;
}

}  // namespace yakimov_i_max_values_in_matrix_rows
