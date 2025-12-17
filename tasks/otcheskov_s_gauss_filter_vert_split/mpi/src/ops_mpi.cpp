#include "otcheskov_s_gauss_filter_vert_split/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "otcheskov_s_gauss_filter_vert_split/common/include/common.hpp"

namespace otcheskov_s_gauss_filter_vert_split {

OtcheskovSGaussFilterVertSplitMPI::OtcheskovSGaussFilterVertSplitMPI(const InType &in) {
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num_);
  SetTypeOfTask(GetStaticTypeOfTask());

  if (proc_rank_ == 0) {
    GetInput() = in;
  }
}

bool OtcheskovSGaussFilterVertSplitMPI::ValidationImpl() {
  bool is_valid = false;
  if (proc_rank_ == 0) {
    const auto &input = GetInput();
    is_valid = !input.data.empty() && input.height >= 3 && input.width >= 3 && input.channels > 0 &&
               (input.data.size() == static_cast<std::size_t>(input.height * input.width * input.channels));
  }
  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return is_valid;
}

bool OtcheskovSGaussFilterVertSplitMPI::PreProcessingImpl() {
  return true;
}

bool OtcheskovSGaussFilterVertSplitMPI::RunImpl() {
  BroadcastImgMetadata();
  // 1. Распределение данных по столбцам
  DistributeData();

  // 2. Обмен граничными столбцами для корректной обработки фильтром 3x3
  ExchangeBoundaryColumns();

  // 3. Применение фильтра Гаусса 3x3
  ApplyGaussianFilter();

  // 4. Сбор результатов от всех процессов
  CollectResults();
  return true;
}

bool OtcheskovSGaussFilterVertSplitMPI::PostProcessingImpl() {
  return true;
}

int OtcheskovSGaussFilterVertSplitMPI::GetGlobalIndex(int row, int col, int channel) {
  return (row * GetInput().width + col) * GetInput().channels + channel;
}

int OtcheskovSGaussFilterVertSplitMPI::GetLocalIndex(int row, int local_col, int channel, int width) {
  return (row * width + local_col) * GetInput().channels + channel;
}
void OtcheskovSGaussFilterVertSplitMPI::BroadcastImgMetadata() {
  int dims[3];
  if (proc_rank_ == 0) {
    dims[0] = GetInput().height;
    dims[1] = GetInput().width;
    dims[2] = GetInput().channels;
  }
  MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);

  if (proc_rank_ != 0) {
    GetInput().height = dims[0];
    GetInput().width = dims[1];
    GetInput().channels = dims[2];
  }
}

int OtcheskovSGaussFilterVertSplitMPI::CalcColsForProcs(int proc_id, int total_width, int num_procs) {
  int base_cols = total_width / num_procs;
  int remainder = total_width % num_procs;
  return base_cols + (proc_id < remainder ? 1 : 0);
}

int OtcheskovSGaussFilterVertSplitMPI::CalcStartCol(int proc_id, int total_width, int num_procs) {
  int start_col = 0;
  for (int i = 0; i < proc_id; ++i) {
    start_col += CalcColsForProcs(i, total_width, num_procs);
  }
  return start_col;
}

void OtcheskovSGaussFilterVertSplitMPI::DistributeData() {
  const auto &input = GetInput();

  // Вычисляем размеры для каждого процесса
  local_width_ = CalcColsForProcs(proc_rank_, input.width, proc_num_);
  start_col_ = CalcStartCol(proc_rank_, input.width, proc_num_);
  local_data_count_ = input.height * local_width_ * input.channels;

  if (proc_rank_ == 0) {
    GetOutput().height = input.height;
    GetOutput().width = input.width;
    GetOutput().channels = input.channels;
    GetOutput().data.resize(input.data.size());

    PrepareAndScatterData();
  } else {
    ReceiveLocalData();
  }
}

void OtcheskovSGaussFilterVertSplitMPI::PrepareAndScatterData() {
  const auto &input = GetInput();
  std::vector<int> counts(proc_num_);
  std::vector<int> displs(proc_num_);
  std::vector<uint8_t> send_buffer;

  int total_send_size = 0;
  for (int p = 0; p < proc_num_; ++p) {
    int cols_for_proc = CalcColsForProcs(p, input.width, proc_num_);
    counts[p] = input.height * cols_for_proc * input.channels;
    displs[p] = total_send_size;
    total_send_size += counts[p];
  }

  send_buffer.resize(total_send_size);
  for (int p = 0; p < proc_num_; ++p) {
    int cols_for_proc = CalcColsForProcs(p, input.width, proc_num_);
    int start_col = CalcStartCol(p, input.width, proc_num_);

    int buf_idx = displs[p];
    for (int i = 0; i < input.height; ++i) {
      for (int j = 0; j < cols_for_proc; ++j) {
        int global_col = start_col + j;
        for (int c = 0; c < input.channels; ++c) {
          send_buffer[buf_idx++] = input.data[GetGlobalIndex(i, global_col, c)];
        }
      }
    }
  }
  // Распределяем данные с использованием Scatterv
  local_data_.resize(local_data_count_);
  MPI_Scatterv(send_buffer.data(), counts.data(), displs.data(), MPI_UINT8_T, local_data_.data(), local_data_count_,
               MPI_INT, 0, MPI_COMM_WORLD);
}

void OtcheskovSGaussFilterVertSplitMPI::ReceiveLocalData() {
  local_data_.resize(local_data_count_);

  // Получаем данные от процесса 0
  MPI_Status status;
  MPI_Recv(local_data_.data(), local_data_count_, MPI_UINT8_T, 0, 0, MPI_COMM_WORLD, &status);
}

void OtcheskovSGaussFilterVertSplitMPI::ExchangeBoundaryColumns() {
  const auto &input = GetInput();
  int height = input.height;
  int channels = input.channels;
  int column_size = height * channels;

  // Определяем соседей
  int left_proc = (proc_rank_ > 0) ? proc_rank_ - 1 : MPI_PROC_NULL;
  int right_proc = (proc_rank_ < proc_num_ - 1) ? proc_rank_ + 1 : MPI_PROC_NULL;

  // Подготавливаем граничные столбцы
  std::vector<uint8_t> left_column(column_size);
  std::vector<uint8_t> right_column(column_size);

  ExtractBoundaryColumns(left_column, right_column);

  // Обмениваемся граничными столбцами
  std::vector<uint8_t> received_left(column_size);
  std::vector<uint8_t> received_right(column_size);

  ExchangeColumnsWithNeighbors(left_column, right_column, received_left, received_right, left_proc, right_proc);

  CreateExtendedData(received_left, received_right);
}

void OtcheskovSGaussFilterVertSplitMPI::ExtractBoundaryColumns(std::vector<uint8_t> &left_column,
                                                               std::vector<uint8_t> &right_column) {
  const auto &input = GetInput();
  int height = input.height;
  int channels = input.channels;

  for (int i = 0; i < height; ++i) {
    for (int c = 0; c < channels; ++c) {
      left_column[i * channels + c] = local_data_[GetLocalIndex(i, 0, c, local_width_)];
      if (local_width_ > 0) {
        right_column[i * channels + c] = local_data_[GetLocalIndex(i, local_width_ - 1, c, local_width_)];
      }
    }
  }
}

void OtcheskovSGaussFilterVertSplitMPI::ExchangeColumnsWithNeighbors(const std::vector<uint8_t> &left_column,
                                                                     const std::vector<uint8_t> &right_column,
                                                                     std::vector<uint8_t> &received_left,
                                                                     std::vector<uint8_t> &received_right,
                                                                     int left_proc, int right_proc) {
  const auto &input = GetInput();
  int column_size = input.height * input.channels;

  MPI_Status status;

  // Обмен с левым соседом
  if (left_proc != MPI_PROC_NULL) {
    MPI_Sendrecv(left_column.data(), column_size, MPI_UINT8_T, left_proc, 0, received_right.data(), column_size,
                 MPI_UINT8_T, left_proc, 1, MPI_COMM_WORLD, &status);
  }

  // Обмен с правым соседом
  if (right_proc != MPI_PROC_NULL) {
    MPI_Sendrecv(right_column.data(), column_size, MPI_UINT8_T, right_proc, 1, received_left.data(), column_size,
                 MPI_UINT8_T, right_proc, 0, MPI_COMM_WORLD, &status);
  }
}

void OtcheskovSGaussFilterVertSplitMPI::CreateExtendedData(const std::vector<uint8_t> &received_left,
                                                           const std::vector<uint8_t> &received_right) {
  const auto &input = GetInput();
  int height = input.height;
  int channels = input.channels;
  int extended_width = local_width_ + 2;

  extended_data_.resize(height * extended_width * channels);

  // Копируем левый граничный столбец
  if (proc_rank_ > 0) {
    for (int i = 0; i < height; ++i) {
      for (int c = 0; c < channels; ++c) {
        extended_data_[GetLocalIndex(i, 0, c, extended_width)] = received_right[i * channels + c];
      }
    }
  } else {
    // Первый процесс использует свой первый столбец как левую границу
    for (int i = 0; i < height; ++i) {
      for (int c = 0; c < channels; ++c) {
        extended_data_[GetLocalIndex(i, 0, c, extended_width)] = local_data_[GetLocalIndex(i, 0, c, local_width_)];
      }
    }
  }

  // Копируем основные данные
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < local_width_; ++j) {
      for (int c = 0; c < channels; ++c) {
        extended_data_[GetLocalIndex(i, j + 1, c, extended_width)] = local_data_[GetLocalIndex(i, j, c, local_width_)];
      }
    }
  }

  // Копируем правый граничный столбец
  if (proc_rank_ < proc_num_ - 1) {
    for (int i = 0; i < height; ++i) {
      for (int c = 0; c < channels; ++c) {
        extended_data_[GetLocalIndex(i, extended_width - 1, c, extended_width)] = received_left[i * channels + c];
      }
    }
  } else if (local_width_ > 0) {
    // Последний процесс использует свой последний столбец как правую границу
    for (int i = 0; i < height; ++i) {
      for (int c = 0; c < channels; ++c) {
        extended_data_[GetLocalIndex(i, extended_width - 1, c, extended_width)] =
            local_data_[GetLocalIndex(i, local_width_ - 1, c, local_width_)];
      }
    }
  }
}

void OtcheskovSGaussFilterVertSplitMPI::ApplyGaussianFilter() {
  const auto &input = GetInput();
  int height = input.height;
  int channels = input.channels;
  int extended_width = local_width_ + 2;

  local_output_.resize(local_data_count_);

  // Применяем фильтр к каждому пикселю
  for (int i = 0; i < height; ++i) {
    for (int local_j = 0; local_j < local_width_; ++local_j) {
      int global_j = start_col_ + local_j;

      for (int c = 0; c < channels; ++c) {
        if (IsBoundaryPixel(i, global_j)) {
          HandleBoundaryPixel(i, local_j, global_j, c);
        } else {
          ApplyGaussianKernel(i, local_j, c, extended_width);
        }
      }
    }
  }
}

bool OtcheskovSGaussFilterVertSplitMPI::IsBoundaryPixel(int row, int col) {
  return row == 0 || row == GetInput().height - 1 || col == 0 || col == GetInput().width - 1;
}

void OtcheskovSGaussFilterVertSplitMPI::HandleBoundaryPixel(int row, int local_j, int global_j, int channel) {
  const auto &input = GetInput();
  int extended_width = local_width_ + 2;
  int ext_j = local_j + 1;

  // Для граничных пикселей просто копируем значение
  if (proc_rank_ == 0) {
    local_output_[GetLocalIndex(row, local_j, channel, local_width_)] =
        input.data[GetGlobalIndex(row, global_j, channel)];
  } else {
    local_output_[GetLocalIndex(row, local_j, channel, local_width_)] =
        extended_data_[GetLocalIndex(row, ext_j, channel, extended_width)];
  }
}

void OtcheskovSGaussFilterVertSplitMPI::ApplyGaussianKernel(int row, int local_j, int channel, int extended_width) {
  double sum = 0.0;
  int ext_j = local_j + 1;  // Смещение в extended_data_

  for (int ki = -1; ki <= 1; ++ki) {
    for (int kj = -1; kj <= 1; ++kj) {
      int data_row = row + ki;
      int data_col = ext_j + kj;

      int data_idx = GetLocalIndex(data_row, data_col, channel, extended_width);
      sum += extended_data_[data_idx] * GAUSSIAN_KERNEL[ki + 1][kj + 1];
    }
  }

  // Ограничиваем значение и сохраняем
  local_output_[GetLocalIndex(row, local_j, channel, local_width_)] = static_cast<uint8_t>(std::clamp(sum, 0.0, 255.0));
}

void OtcheskovSGaussFilterVertSplitMPI::CollectResults() {
  if (proc_rank_ == 0) {
    // Процесс 0 собирает данные от всех процессов
    GatherResultsFromAllProcesses();
  } else {
    // Отправляем свои данные процессу 0
    SendLocalResults();
  }
}

void OtcheskovSGaussFilterVertSplitMPI::GatherResultsFromAllProcesses() {
  // Сначала копируем свои данные
  CopyLocalResultsToOutput();

  // Принимаем данные от других процессов
  for (int p = 1; p < proc_num_; ++p) {
    ReceiveAndStoreProcessResults(p);
  }
}

void OtcheskovSGaussFilterVertSplitMPI::CopyLocalResultsToOutput() {
  const auto &input = GetInput();

  for (int i = 0; i < input.height; ++i) {
    for (int local_j = 0; local_j < local_width_; ++local_j) {
      int global_j = start_col_ + local_j;
      for (int c = 0; c < input.channels; ++c) {
        GetOutput().data[GetGlobalIndex(i, global_j, c)] = local_output_[GetLocalIndex(i, local_j, c, local_width_)];
      }
    }
  }
}

void OtcheskovSGaussFilterVertSplitMPI::ReceiveAndStoreProcessResults(int proc_id) {
  const auto &input = GetInput();

  int cols_for_proc = CalcColsForProcs(proc_id, input.width, proc_num_);
  int proc_data_count = input.height * cols_for_proc * input.channels;

  std::vector<uint8_t> proc_data(proc_data_count);
  MPI_Status status;
  MPI_Recv(proc_data.data(), proc_data_count, MPI_UINT8_T, proc_id, 0, MPI_COMM_WORLD, &status);

  int start_col = CalcStartCol(proc_id, input.width, proc_num_);
  int buf_idx = 0;

  for (int i = 0; i < input.height; ++i) {
    for (int j = 0; j < cols_for_proc; ++j) {
      int global_col = start_col + j;
      for (int c = 0; c < input.channels; ++c) {
        GetOutput().data[GetGlobalIndex(i, global_col, c)] = proc_data[buf_idx++];
      }
    }
  }
}

void OtcheskovSGaussFilterVertSplitMPI::SendLocalResults() {
  MPI_Send(local_output_.data(), local_data_count_, MPI_UINT8_T, 0, 0, MPI_COMM_WORLD);
}

}  // namespace otcheskov_s_gauss_filter_vert_split
