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
    channels_ = in.channels;
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
  if (is_valid) {
    MPI_Bcast(&channels_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
  return is_valid;
}

bool OtcheskovSGaussFilterVertSplitMPI::PreProcessingImpl() {
  return true;
}

bool OtcheskovSGaussFilterVertSplitMPI::RunImpl() {
  bool is_valid = false;
  if (proc_rank_ == 0) {
    const auto &input = GetInput();
    is_valid = !input.data.empty() && input.height >= 3 && input.width >= 3 && input.channels > 0 &&
               (input.data.size() == static_cast<std::size_t>(input.height * input.width * input.channels));
  }
  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  if (!is_valid) {
    return false;
  }
  BroadcastImgMetadata();
  DistributeData();
  ExchangeBoundaryColumns();

  local_data_.clear();
  local_data_.shrink_to_fit();

  ApplyGaussianFilter();
  CollectResults();
  return true;
}

bool OtcheskovSGaussFilterVertSplitMPI::PostProcessingImpl() {
  return true;
}

void OtcheskovSGaussFilterVertSplitMPI::BroadcastImgMetadata() {
  std::array<int, 3> metadata;
  auto &input = GetInput();
  if (proc_rank_ == 0) {
    metadata = {input.height, input.width, input.channels};
  }
  MPI_Bcast(metadata.data(), metadata.size(), MPI_INT, 0, MPI_COMM_WORLD);

  input.height = metadata[0];
  input.width = metadata[1];
  input.channels = metadata[2];
  channels_ = metadata[2];
}

void OtcheskovSGaussFilterVertSplitMPI::DistributeData() {
  const auto &input = GetInput();
  active_procs_ = std::min(proc_num_, input.width);

  if (proc_rank_ < active_procs_) {
    const int base_cols = input.width / active_procs_;
    const int remainder = input.width % active_procs_;
    local_width_ = base_cols + (proc_rank_ < remainder);
    start_col_ = base_cols * proc_rank_ + std::min(proc_rank_, remainder);
  } else {
    local_width_ = 0;
    start_col_ = 0;
  }

  local_data_count_ = input.height * local_width_ * channels_;

  if (proc_rank_ == 0) {
    auto &output = GetOutput();
    output = ImageData{std::vector<uint8_t>(input.data.size()), input.height, input.width, input.channels};
  }
  local_data_.resize(local_data_count_);

  if (proc_rank_ == 0) {
    const int row_size = input.width * channels_;
    std::vector<int> counts(proc_num_, 0);
    std::vector<int> displs(proc_num_, 0);
    const int base_cols = input.width / active_procs_;
    const int remainder = input.width % active_procs_;

    int total_data = 0;
    for (int p = 0; p < active_procs_; ++p) {
      const int cols = base_cols + (p < remainder);
      counts[p] = input.height * cols * channels_;
      displs[p] = total_data;
      total_data += counts[p];
    }

    std::vector<uint8_t> send_buffer(total_data);
    for (int p = 0; p < active_procs_; ++p) {
      const int cols = base_cols + (p < remainder);
      const int start_col = base_cols * p + std::min(p, remainder);
      uint8_t *buf_ptr = send_buffer.data() + displs[p];

      for (int i = 0; i < input.height; ++i) {
        const uint8_t *src_row = input.data.data() + i * row_size + start_col * channels_;
        std::memcpy(buf_ptr, src_row, cols * channels_);
        buf_ptr += cols * channels_;
      }
    }

    MPI_Scatterv(send_buffer.data(), counts.data(), displs.data(), MPI_UINT8_T, local_data_.data(), local_data_count_,
                 MPI_UINT8_T, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DATATYPE_NULL, local_data_.data(), local_data_count_, MPI_UINT8_T, 0,
                 MPI_COMM_WORLD);
  }
}

void OtcheskovSGaussFilterVertSplitMPI::ExchangeBoundaryColumns() {
  if (local_width_ == 0) {
    extended_data_.clear();
    extended_data_.shrink_to_fit();
    return;
  }

  const auto &input = GetInput();
  const int col_size = input.height * channels_;

  int left_proc = MPI_PROC_NULL;
  int right_proc = MPI_PROC_NULL;

  if (proc_rank_ > 0 && proc_rank_ < active_procs_) {
    left_proc = proc_rank_ - 1;
  }
  if (proc_rank_ < active_procs_ - 1) {
    right_proc = proc_rank_ + 1;
  }

  std::vector<uint8_t> left_col(col_size), right_col(col_size);
  std::vector<uint8_t> recv_left(col_size), recv_right(col_size);

  for (int i = 0; i < input.height; ++i) {
    const int row_off = i * local_width_ * channels_;
    std::memcpy(&left_col[i * channels_], &local_data_[row_off], channels_);
    std::memcpy(&right_col[i * channels_], &local_data_[row_off + (local_width_ - 1) * channels_], channels_);
  }

  MPI_Status status;
  if (left_proc != MPI_PROC_NULL) {
    MPI_Sendrecv(left_col.data(), col_size, MPI_UINT8_T, left_proc, 0, recv_right.data(), col_size, MPI_UINT8_T,
                 left_proc, 1, MPI_COMM_WORLD, &status);
  }
  if (right_proc != MPI_PROC_NULL) {
    MPI_Sendrecv(right_col.data(), col_size, MPI_UINT8_T, right_proc, 1, recv_left.data(), col_size, MPI_UINT8_T,
                 right_proc, 0, MPI_COMM_WORLD, &status);
  }

  const int ext_width = local_width_ + 2;
  extended_data_.resize(input.height * ext_width * channels_);

  for (int i = 0; i < input.height; ++i) {
    uint8_t *ext_row = &extended_data_[i * ext_width * channels_];
    const uint8_t *loc_row = &local_data_[i * local_width_ * channels_];

    if (proc_rank_ == 0) {
      std::memcpy(ext_row, loc_row, channels_);
    } else {
      std::memcpy(ext_row, &recv_right[i * channels_], channels_);
    }

    std::memcpy(ext_row + channels_, loc_row, local_width_ * channels_);

    if (proc_rank_ == active_procs_ - 1) {
      const uint8_t *last_col = &loc_row[(local_width_ - 1) * channels_];
      std::memcpy(ext_row + (ext_width - 1) * channels_, last_col, channels_);
    } else {
      std::memcpy(ext_row + (ext_width - 1) * channels_, &recv_left[i * channels_], channels_);
    }
  }
}

void OtcheskovSGaussFilterVertSplitMPI::ApplyGaussianFilter() {
  if (local_width_ == 0) {
    local_output_.clear();
    local_output_.shrink_to_fit();
    return;
  }

  const auto &input = GetInput();
  const int height = input.height;
  const int extended_width = local_width_ + 2;

  local_output_.resize(local_data_count_);

  for (int i = 0; i < height; ++i) {
    for (int local_j = 0; local_j < local_width_; ++local_j) {
      const int ext_j = local_j + 1;

      for (int c = 0; c < channels_; ++c) {
        double sum = 0.0;

        for (int ki = -1; ki <= 1; ++ki) {
          const int data_row = MirrorRow(i + ki, height);
          for (int kj = -1; kj <= 1; ++kj) {
            const int data_col = ext_j + kj;
            const int idx = GetLocalIndex(data_row, data_col, c, extended_width);
            sum += extended_data_[idx] * kGaussianKernel[ki + 1][kj + 1];
          }
        }

        local_output_[GetLocalIndex(i, local_j, c, local_width_)] =
            static_cast<uint8_t>(std::clamp(std::round(sum), 0.0, 255.0));
      }
    }
  }
}

int OtcheskovSGaussFilterVertSplitMPI::MirrorRow(int row, int height) {
  if (row < 0) {
    row = -row - 1;
  }
  if (row >= height) {
    row = 2 * height - row - 1;
  }
  return row;
}

int OtcheskovSGaussFilterVertSplitMPI::GetLocalIndex(int row, int local_col, int channel, int width) {
  return (row * width + local_col) * channels_ + channel;
}

void OtcheskovSGaussFilterVertSplitMPI::CollectResults() {
  const auto &input = GetInput();

  const int base_cols = input.width / active_procs_;
  const int remainder = input.width % active_procs_;
  const int row_size = input.width * channels_;

  std::vector<int> counts(proc_num_, 0);
  std::vector<int> displs(proc_num_, 0);

  int total_data = 0;
  for (int p = 0; p < active_procs_; ++p) {
    const int cols = base_cols + (p < remainder);
    counts[p] = input.height * cols * channels_;
    displs[p] = total_data;
    total_data += counts[p];
  }

  if (proc_rank_ == 0) {
    std::vector<uint8_t> recv_buffer(total_data);
    MPI_Gatherv(local_output_.data(), local_data_count_, MPI_UINT8_T, recv_buffer.data(), counts.data(), displs.data(),
                MPI_UINT8_T, 0, MPI_COMM_WORLD);

    for (int i = 0; i < input.height; ++i) {
      int buffer_offset = 0;
      for (int p = 0; p < active_procs_; ++p) {
        const int cols = base_cols + (p < remainder);
        const int start_col = base_cols * p + std::min(p, remainder);
        const uint8_t *src = recv_buffer.data() + buffer_offset + i * cols * channels_;
        uint8_t *dst = GetOutput().data.data() + i * row_size + start_col * channels_;
        std::memcpy(dst, src, cols * channels_);

        buffer_offset += counts[p];
      }
    }
  } else {
    MPI_Gatherv(local_output_.data(), local_data_count_, MPI_UINT8_T, nullptr, nullptr, nullptr, MPI_UINT8_T, 0,
                MPI_COMM_WORLD);
  }
}

}  // namespace otcheskov_s_gauss_filter_vert_split
