#include "otcheskov_s_gauss_filter_vert_split/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "otcheskov_s_gauss_filter_vert_split/common/include/common.hpp"

namespace otcheskov_s_gauss_filter_vert_split {

OtcheskovSGaussFilterVertSplitMPI::OtcheskovSGaussFilterVertSplitMPI(const InType &in) {
  int proc_rank{};
  int proc_num{};
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
  SetTypeOfTask(GetStaticTypeOfTask());

  proc_rank_ = static_cast<size_t>(proc_rank);
  proc_num_ = static_cast<size_t>(proc_num);
  if (proc_rank_ == 0) {
    GetInput() = in;
  }
}

bool OtcheskovSGaussFilterVertSplitMPI::ValidationImpl() {
  if (proc_rank_ == 0) {
    const auto &[metadata, data] = GetInput();
    is_valid_ = !data.empty() && (metadata.height > 0 && metadata.width > 0 && metadata.channels > 0) &&
                (data.size() == metadata.height * metadata.width * metadata.channels);
  }
  MPI_Bcast(&is_valid_, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return is_valid_;
}

bool OtcheskovSGaussFilterVertSplitMPI::PreProcessingImpl() {
  return true;
}

bool OtcheskovSGaussFilterVertSplitMPI::RunImpl() {
  if (!is_valid_) {
    return false;
  }

  auto &[metadata, data] = GetInput();
  MPI_Bcast(&metadata, sizeof(ImageMetadata), MPI_BYTE, 0, MPI_COMM_WORLD);

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

void OtcheskovSGaussFilterVertSplitMPI::DistributeData() {
  const auto &[in_meta, in_data] = GetInput();
  const auto &[height, width, channels] = in_meta;

  active_procs_ = std::min(proc_num_, width);

  if (proc_rank_ < active_procs_) {
    const size_t base_cols = width / active_procs_;
    const size_t remain = width % active_procs_;
    local_width_ = base_cols + (proc_rank_ < remain ? 1 : 0);
    start_col_ = (base_cols * proc_rank_) + std::min(proc_rank_, remain);
  } else {
    local_width_ = 0;
    start_col_ = 0;
  }

  local_data_count_ = height * local_width_ * channels;
  local_data_.resize(local_data_count_);

  if (proc_rank_ == 0) {
    const size_t base_cols = width / active_procs_;
    const size_t remain = width % active_procs_;
    const auto &[counts, displs] = GetCountsAndDisplacements(height, width, channels);

    std::vector<uint8_t> send_buffer(counts[active_procs_ - 1] + displs[active_procs_ - 1]);
    for (size_t proc = 0; proc < active_procs_; ++proc) {
      const size_t cols = base_cols + (proc < remain ? 1 : 0);
      const size_t start_col = (base_cols * proc) + std::min(proc, remain);
      uint8_t *buf_ptr = send_buffer.data() + displs[proc];

      for (size_t row = 0; row < height; ++row) {
        const size_t row_size = width * channels;
        const size_t row_offset = row * row_size;
        const size_t col_offset = start_col * channels;
        const uint8_t *src_row = in_data.data() + row_offset + col_offset;
        std::memcpy(buf_ptr, src_row, cols * channels);
        buf_ptr += cols * channels;
      }
    }

    MPI_Scatterv(send_buffer.data(), counts.data(), displs.data(), MPI_UINT8_T, local_data_.data(),
                 static_cast<int>(local_data_count_), MPI_UINT8_T, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DATATYPE_NULL, local_data_.data(), static_cast<int>(local_data_count_),
                 MPI_UINT8_T, 0, MPI_COMM_WORLD);
  }
}

std::pair<std::vector<int>, std::vector<int>> OtcheskovSGaussFilterVertSplitMPI::GetCountsAndDisplacements(
    const size_t &height, const size_t &width, const size_t &channels) const {
  std::vector<int> counts(proc_num_, 0);
  std::vector<int> displs(proc_num_, 0);

  const size_t base_cols = width / active_procs_;
  const size_t remain = width % active_procs_;

  int total_data = 0;
  for (size_t proc = 0; proc < active_procs_; ++proc) {
    const size_t cols = base_cols + (proc < remain ? 1 : 0);
    counts[proc] = static_cast<int>(height * cols * channels);
    displs[proc] = total_data;
    total_data += counts[proc];
  }

  return {counts, displs};
}

void OtcheskovSGaussFilterVertSplitMPI::ExchangeBoundaryColumns() {
  if (local_width_ == 0) {
    extended_data_.clear();
    extended_data_.shrink_to_fit();
    return;
  }

  const auto &[in_meta, in_data] = GetInput();
  const size_t &height = in_meta.height;
  const size_t &channels = in_meta.channels;
  const size_t col_size = height * channels;

  int left_proc = MPI_PROC_NULL;
  int right_proc = MPI_PROC_NULL;

  if (proc_rank_ > 0 && proc_rank_ < active_procs_) {
    left_proc = static_cast<int>(proc_rank_) - 1;
  }
  if (proc_rank_ < active_procs_ - 1) {
    right_proc = static_cast<int>(proc_rank_) + 1;
  }

  std::vector<uint8_t> left_col(col_size);
  std::vector<uint8_t> right_col(col_size);
  std::vector<uint8_t> recv_left(col_size);
  std::vector<uint8_t> recv_right(col_size);

  for (size_t i = 0; i < height; ++i) {
    const size_t row_off = i * local_width_ * channels;
    const size_t dst_offset = i * channels;

    std::memcpy(&left_col[dst_offset], &local_data_[row_off], channels);
    std::memcpy(&right_col[dst_offset], &local_data_[row_off + ((local_width_ - 1) * channels)], channels);
  }

  if (left_proc != MPI_PROC_NULL) {
    MPI_Sendrecv(left_col.data(), static_cast<int>(col_size), MPI_UINT8_T, left_proc, 0, recv_right.data(),
                 static_cast<int>(col_size), MPI_UINT8_T, left_proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  if (right_proc != MPI_PROC_NULL) {
    MPI_Sendrecv(right_col.data(), static_cast<int>(col_size), MPI_UINT8_T, right_proc, 1, recv_left.data(),
                 static_cast<int>(col_size), MPI_UINT8_T, right_proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  const size_t ext_width = local_width_ + 2;
  extended_data_.resize(in_meta.height * ext_width * channels);

  for (size_t i = 0; i < in_meta.height; ++i) {
    uint8_t *ext_row = &extended_data_[i * ext_width * channels];
    const uint8_t *loc_row = &local_data_[i * local_width_ * channels];

    if (proc_rank_ == 0) {
      std::memcpy(ext_row, loc_row, channels);
    } else {
      std::memcpy(ext_row, &recv_right[i * channels], channels);
    }

    std::memcpy(ext_row + channels, loc_row, local_width_ * channels);

    if (proc_rank_ == active_procs_ - 1) {
      const uint8_t *last_col = &loc_row[(local_width_ - 1) * channels];
      std::memcpy(ext_row + ((ext_width - 1) * channels), last_col, channels);
    } else {
      std::memcpy(ext_row + ((ext_width - 1) * channels), &recv_left[i * channels], channels);
    }
  }
}

void OtcheskovSGaussFilterVertSplitMPI::ApplyGaussianFilter() {
  if (local_width_ == 0) {
    return;
  }
  const auto &[in_meta, in_data] = GetInput();
  local_output_.resize(local_data_count_);

  for (size_t row = 0; row < in_meta.height; ++row) {
    for (size_t local_col = 0; local_col < local_width_; ++local_col) {
      for (size_t ch = 0; ch < in_meta.channels; ++ch) {
        const size_t out_idx = ((row * local_width_ + local_col) * in_meta.channels) + ch;
        local_output_[out_idx] = ProcessPixel(row, local_col, ch, in_meta.height, in_meta.channels);
      }
    }
  }
}

uint8_t OtcheskovSGaussFilterVertSplitMPI::ProcessPixel(const size_t &row, const size_t &local_col, const size_t &ch,
                                                        const size_t &height, const size_t &channels) {
  auto mirror_coord = [&](const size_t &current, int off, const size_t &size) -> size_t {
    int64_t pos = static_cast<int64_t>(current) + off;
    if (pos < 0) {
      return static_cast<size_t>(-pos - 1);
    }
    if (std::cmp_greater_equal(static_cast<size_t>(pos), size)) {
      return (2 * size) - static_cast<size_t>(pos) - 1;
    }
    return static_cast<size_t>(pos);
  };

  double sum = 0.0;
  const size_t extended_width = local_width_ + 2;
  const size_t ext_col = local_col + 1;

  for (int ky = 0; ky < 3; ++ky) {
    const size_t data_row = mirror_coord(row, ky - 1, height);

    for (int kx = 0; kx < 3; ++kx) {
      const size_t data_col = ext_col + kx - 1;
      const size_t idx = ((data_row * extended_width + data_col) * channels) + ch;
      sum += extended_data_[idx] * kGaussianKernel.at(ky).at(kx);
    }
  }
  return static_cast<uint8_t>(std::clamp(std::round(sum), 0.0, 255.0));
}

void OtcheskovSGaussFilterVertSplitMPI::CollectResults() {
  const auto &[in_meta, in_data] = GetInput();
  const auto &[height, width, channels] = in_meta;
  const size_t base_cols = width / active_procs_;
  const size_t remain = width % active_procs_;
  const size_t row_size = width * channels;

  std::vector<int> counts(proc_num_, 0);
  std::vector<int> displs(proc_num_, 0);

  int total_data = 0;
  for (size_t proc = 0; proc < active_procs_; ++proc) {
    const size_t cols = base_cols + (proc < remain ? 1 : 0);
    counts[proc] = static_cast<int>(height * cols * channels);
    displs[proc] = total_data;
    total_data += counts[proc];
  }

  if (proc_rank_ == 0) {
    auto &[out_meta, out_data] = GetOutput();
    out_meta = in_meta;
    out_data.resize(in_data.size());

    std::vector<uint8_t> recv_buffer(total_data);
    MPI_Gatherv(local_output_.data(), static_cast<int>(local_data_count_), MPI_UINT8_T, recv_buffer.data(),
                counts.data(), displs.data(), MPI_UINT8_T, 0, MPI_COMM_WORLD);

    for (size_t i = 0; i < height; ++i) {
      int buffer_offset = 0;
      for (size_t proc = 0; proc < active_procs_; ++proc) {
        const size_t cols = base_cols + (proc < remain ? 1 : 0);
        const size_t start_col = (base_cols * proc) + std::min(proc, remain);
        const uint8_t *src = recv_buffer.data() + static_cast<size_t>(buffer_offset) + (i * cols * channels);
        uint8_t *dst = out_data.data() + (i * row_size) + (start_col * channels);
        std::memcpy(dst, src, cols * channels);

        buffer_offset += counts[proc];
      }
    }
  } else {
    MPI_Gatherv(local_output_.data(), static_cast<int>(local_data_count_), MPI_UINT8_T, nullptr, nullptr, nullptr,
                MPI_UINT8_T, 0, MPI_COMM_WORLD);
  }
}

}  // namespace otcheskov_s_gauss_filter_vert_split
