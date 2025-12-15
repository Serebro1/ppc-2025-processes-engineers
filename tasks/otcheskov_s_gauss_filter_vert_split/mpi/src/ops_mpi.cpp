#include "otcheskov_s_gauss_filter_vert_split/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "ops_mpi.hpp"
#include "otcheskov_s_gauss_filter_vert_split/common/include/common.hpp"

namespace otcheskov_s_elem_vec_avg {

otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitMPI::OtcheskovSGaussFilterVertSplitMPI(
    const InType &in) {
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num_);
  SetTypeOfTask(GetStaticTypeOfTask());

  if (proc_rank_ == 0) {
    GetInput() = in;
  }
}

bool otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitMPI::ValidationImpl() {
  bool is_valid = false;
  if (proc_rank_ == 0) {
    const auto &input = GetInput();
    is_valid = input.data.empty() && (input.height < 3 || input.width < 3 || input.channels <= 0) &&
               (input.data.size() != static_cast<std::size_t>(input.height * input.width * input.channels));
  }
  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return is_valid;
}

bool otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitMPI::PreProcessingImpl() {
  return true;
}

bool otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitMPI::RunImpl() {
  // 1. Распределение данных по столбцам
  distributeData();

  // 2. Обмен граничными столбцами для корректной обработки фильтром 3x3
  exchangeBoundaryColumns();

  // 3. Применение фильтра Гаусса 3x3
  applyGaussianFilter();

  // 4. Сбор результатов от всех процессов
  collectResults();
  return true;
}

bool otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitMPI::PostProcessingImpl() {
  return true;
}

int otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitMPI::GetGlobalIndex(int row, int col,
                                                                                           int channel) const {
  return (row * GetInput().width + col) * GetInput().channels + channel;
}
int otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitMPI::GetLocalIndex(int row, int local_col,
                                                                                          int channel) const {
  return (row * local_width_ + local_col) * GetInput().channels + channel;
}
void otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitMPI::DistributeData(
    std::vector<int> &local_data) {
  const auto &input = GetInput();

  int base_cols = input.width / proc_num_;
  int remainder = input.width % proc_num_;

  local_width_ = base_cols + (proc_rank_ < remainder ? 1 : 0);
  local_data_count_ = input.height * local_width_ * input.channels;

  start_col_ = 0;
  for (int i = 0; i < proc_rank_; ++i) {
    int cols_for_proc = base_cols + (i < remainder ? 1 : 0);
    start_col_ += cols_for_proc;
  }

  if (proc_rank_ == 0) {
    GetOutput().height = input.height;
    GetOutput().width = input.width;
    GetOutput().channels = input.channels;
    GetOutput().data.resize(input.data.size());

    std::vector<std::vector<uint8_t>> send_buffers(proc_num_);
    std::vector<uint8_t> sendcounts(proc_num_, 0);
    int base_cols = input.width / proc_num_;
    int remainder = input.width % proc_num_;
    int cols_for_proc = base_cols + (p < remainder ? 1 : 0);
    int start_col = 0;

    for (int p = 0; p < proc_num_; ++p) {
      sendcounts[p] = input.height * cols_for_proc * input.channels;
      send_buffers[p].resize(sendcounts[p]);
    }

    for (int p = 0; p < proc_num_; ++p) {
      // Вычисляем начальный столбец для процесса p
      for (int i = 0; i < p; ++i) {
        int proc_cols = base_cols + (i < remainder ? 1 : 0);
        start_col += proc_cols;
      }

      // Копируем данные по столбцам
      int buf_idx = 0;
      for (int i = 0; i < input.height; ++i) {
        for (int j = 0; j < cols_for_proc; ++j) {
          int global_col = start_col + j;
          for (int c = 0; c < input.channels; ++c) {
            send_buffers[p][buf_idx++] = input.data[getGlobalIndex(i, global_col, c)];
          }
        }
      }
    }
    for (int p = 1; p < proc_num_; ++p) {
      MPI_Send(send_buffers[p].data(), sendcounts[p], MPI_INT, p, 0, MPI_COMM_WORLD);
    }

    local_data_ = std::move(send_buffers[0]);
  } else {
    local_data_.resize(local_data_count_);
    MPI_Recv(local_data_.data(), local_data_count_, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitMPI::ExchangeBoundaryRows(
    std::vector<int> &local_data) {
  const auto &input = GetInput();
  int row_size_with_channels = local_width_ * input.channels;
  int height = input.height;

  int left_proc = proc_rank_ - 1;
  int right_proc = proc_rank_ + 1;

  if (proc_rank_ == 0) {
    left_proc = MPI_PROC_NULL;
  }
  if (proc_rank_ == proc_num_ - 1) {
    right_proc = MPI_PROC_NULL;
  }

  std::vector<int> left_column(height * input.channels);
  std::vector<int> right_column(height * input.channels);

  // Извлекаем левый и правый граничные столбцы из локальных данных
  for (int i = 0; i < height; ++i) {
    for (int c = 0; c < input.channels; ++c) {
      // Левый граничный столбец (первый столбец в локальных данных)
      left_column[i * input.channels + c] = local_data_[getLocalIndex(i, 0, c)];

      // Правый граничный столбец (последний столбец в локальных данных)
      right_column[i * input.channels + c] = local_data_[getLocalIndex(i, local_width_ - 1, c)];
    }
  }

  // Буферы для получения граничных столбцов от соседей
  std::vector<uint8_t> received_left_column(height * input.channels);
  std::vector<uint8_t> received_right_column(height * input.channels);

  // Обмен граничными столбцами
  // Отправляем левый столбец левому соседу, получаем от него правый столбец
  if (left_proc != MPI_PROC_NULL) {
    MPI_Sendrecv(left_column.data(), height * input.channels, MPI_INT, left_proc, 0, received_right_column.data(),
                 height * input.channels, MPI_INT, left_proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Отправляем правый столбец правому соседу, получаем от него левый столбец
  if (right_proc != MPI_PROC_NULL) {
    MPI_Sendrecv(right_column.data(), height * input.channels, MPI_INT, right_proc, 1, received_left_column.data(),
                 height * input.channels, MPI_INT, right_proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Создаем расширенные данные с двумя дополнительными столбцами (слева и справа)
  int extended_width = local_width_ + 2;
  extended_data_.resize(height * extended_width * input.channels);

  // Копируем левый граничный столбец (если есть)
  if (left_proc != MPI_PROC_NULL) {
    for (int i = 0; i < height; ++i) {
      for (int c = 0; c < input.channels; ++c) {
        extended_data_[getLocalIndex(i, 0, c)] = received_right_column[i * input.channels + c];
      }
    }
  } else if (start_col_ > 0) {
    // Если это первый столбец изображения, копируем из своих данных
    for (int i = 0; i < height; ++i) {
      for (int c = 0; c < input.channels; ++c) {
        extended_data_[getLocalIndex(i, 0, c)] = local_data_[getLocalIndex(i, 0, c)];
      }
    }
  }

  // Копируем основные данные (сдвиг на 1, т.к. есть левый граничный столбец)
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < local_width_; ++j) {
      for (int c = 0; c < input.channels; ++c) {
        extended_data_[getLocalIndex(i, j + 1, c)] = local_data_[getLocalIndex(i, j, c)];
      }
    }
  }

  // Копируем правый граничный столбец (если есть)
  if (right_proc != MPI_PROC_NULL) {
    for (int i = 0; i < height; ++i) {
      for (int c = 0; c < input.channels; ++c) {
        extended_data_[getLocalIndex(i, extended_width - 1, c)] = received_left_column[i * input.channels + c];
      }
    }
  } else if (start_col_ + local_width_ < input.width) {
    // Если это последний столбец изображения, копируем из своих данных
    for (int i = 0; i < height; ++i) {
      for (int c = 0; c < input.channels; ++c) {
        extended_data_[getLocalIndex(i, extended_width - 1, c)] = local_data_[getLocalIndex(i, local_width_ - 1, c)];
      }
    }
  }
}

void otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitMPI::ApplyGaussianFilter(
    const std::vector<int> &ext_data, std::vector<int> &local_output) {
  const auto &input = GetInput();
  int height = input.height;
  int extended_width = local_width_ + 2;
  int channels = input.channels;

  local_output_.resize(local_data_count_);

  // Применяем фильтр Гаусса 3x3 к локальным данным
  for (int i = 0; i < height; ++i) {
    for (int local_j = 0; local_j < local_width_; ++local_j) {
      int global_j = start_col_ + local_j;
      int ext_j = local_j + 1;  // В расширенных данных (т.к. есть левый граничный столбец)

      for (int c = 0; c < channels; ++c) {
        // Проверяем, не граничный ли это пиксель
        bool is_boundary = (i == 0 || i == height - 1 || global_j == 0 || global_j == input.width - 1);

        if (is_boundary) {
          // Для граничных пикселей просто копируем значение
          if (proc_rank_ == 0) {
            // Процесс 0 имеет доступ к исходным данным
            local_output_[getLocalIndex(i, local_j, c)] = GetInput().data[getGlobalIndex(i, global_j, c)];
          } else {
            // Для других процессов значение берется из расширенных данных
            local_output_[getLocalIndex(i, local_j, c)] = extended_data_[getLocalIndex(i, ext_j, c)];
          }
        } else {
          // Применяем полноценный фильтр Гаусса 3x3
          double sum = 0.0;

          // Двумерная свертка с ядром Гаусса 3x3
          for (int ki = -1; ki <= 1; ++ki) {
            for (int kj = -1; kj <= 1; ++kj) {
              int data_row = i + ki;
              int data_col = ext_j + kj;

              int data_idx = getLocalIndex(data_row, data_col, c);
              sum += extended_data_[data_idx] * GAUSSIAN_KERNEL[ki + 1][kj + 1];
            }
          }

          // Ограничиваем значение и записываем
          int value = static_cast<int>(std::round(sum));
          value = std::max(0, std::min(255, value));
          local_output_[getLocalIndex(i, local_j, c)] = value;
        }
      }
    }
  }

  // Специальная обработка для процесса 0 - граничные столбцы всего изображения
  if (proc_rank_ == 0 && start_col_ == 0) {
    // Первый столбец изображения (левый граничный)
    for (int i = 0; i < height; ++i) {
      for (int c = 0; c < channels; ++c) {
        local_output_[getLocalIndex(i, 0, c)] = GetInput().data[getGlobalIndex(i, 0, c)];
      }
    }
  }

  // Специальная обработка для последнего процесса - правый граничный столбец
  if (proc_rank_ == proc_num_ - 1) {
    int last_local_col = local_width_ - 1;
    int global_last_col = input.width - 1;

    if (start_col_ + local_width_ == input.width) {
      for (int i = 0; i < height; ++i) {
        for (int c = 0; c < channels; ++c) {
          local_output_[getLocalIndex(i, last_local_col, c)] = GetInput().data[getGlobalIndex(i, global_last_col, c)];
        }
      }
    }
  }
}

void otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitMPI::CollectResults(
    const std::vector<int> &local_output) {
  const auto &input = GetInput();

  if (proc_rank_ == 0) {
    // Процесс 0 собирает данные от всех процессов

    // Сначала копируем свои данные
    for (int i = 0; i < input.height; ++i) {
      for (int local_j = 0; local_j < local_width_; ++local_j) {
        int global_j = start_col_ + local_j;
        for (int c = 0; c < input.channels; ++c) {
          GetOutput().data[getGlobalIndex(i, global_j, c)] = local_output_[getLocalIndex(i, local_j, c)];
        }
      }
    }

    // Принимаем данные от других процессов
    for (int p = 1; p < proc_num_; ++p) {
      // Получаем размер данных от процесса p
      int base_cols = input.width / proc_num_;
      int remainder = input.width % proc_num_;
      int cols_for_proc = base_cols + (p < remainder ? 1 : 0);
      int proc_data_count = input.height * cols_for_proc * input.channels;

      std::vector<int> proc_data(proc_data_count);
      MPI_Recv(proc_data.data(), proc_data_count, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // Определяем начальный столбец для процесса p
      int start_col = 0;
      for (int i = 0; i < p; ++i) {
        int proc_cols = base_cols + (i < remainder ? 1 : 0);
        start_col += proc_cols;
      }

      // Копируем данные процесса p в выходной массив
      int buf_idx = 0;
      for (int i = 0; i < input.height; ++i) {
        for (int j = 0; j < cols_for_proc; ++j) {
          int global_col = start_col + j;
          for (int c = 0; c < input.channels; ++c) {
            GetOutput().data[getGlobalIndex(i, global_col, c)] = proc_data[buf_idx++];
          }
        }
      }
    }
  } else {
    // Отправляем свои данные процессу 0
    MPI_Send(local_output_.data(), local_data_count_, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
}

}  // namespace otcheskov_s_elem_vec_avg
