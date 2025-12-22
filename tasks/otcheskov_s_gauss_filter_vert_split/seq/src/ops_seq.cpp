#include "otcheskov_s_gauss_filter_vert_split/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "otcheskov_s_gauss_filter_vert_split/common/include/common.hpp"

namespace otcheskov_s_gauss_filter_vert_split {

OtcheskovSGaussFilterVertSplitSEQ::OtcheskovSGaussFilterVertSplitSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool OtcheskovSGaussFilterVertSplitSEQ::ValidationImpl() {
  const auto &[metadata, data] = GetInput();

  is_valid_ = !data.empty() && (metadata.height > 0 && metadata.width > 0 && metadata.channels > 0) &&
              (data.size() == metadata.height * metadata.width * metadata.channels);

  return is_valid_;
}

bool OtcheskovSGaussFilterVertSplitSEQ::PreProcessingImpl() {
  return true;
}

bool OtcheskovSGaussFilterVertSplitSEQ::RunImpl() {
  if (!is_valid_) {
    return false;
  }

  const auto &[in_meta, in_data] = GetInput();
  auto &[out_meta, out_data] = GetOutput();
  out_meta = in_meta;
  out_data.resize(in_data.size());
  const auto &[height, width, channels] = in_meta;

  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      for (size_t ch = 0; ch < channels; ++ch) {
        size_t out_idx = (((row * width) + col) * channels) + ch;
        out_data[out_idx] = ProcessPixel(row, col, ch);
      }
    }
  }
  return true;
}

uint8_t OtcheskovSGaussFilterVertSplitSEQ::ProcessPixel(size_t row, size_t col, size_t ch) {
  const auto &[in_meta, in_data] = GetInput();
  const auto &[height, width, channels] = in_meta;

  auto mirror_coord = [&](const size_t &current, const int &off, const size_t &size) -> size_t {
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
  for (int dy = 0; dy < 3; ++dy) {
    size_t src_y = mirror_coord(row, dy - 1, height);
    for (int dx = 0; dx < 3; ++dx) {
      size_t src_x = mirror_coord(col, dx - 1, width);
      double weight = kGaussianKernel.at(dy).at(dx);
      size_t src_idx = (((src_y * width) + src_x) * channels) + ch;
      sum += weight * in_data[src_idx];
    }
  }
  return static_cast<uint8_t>(std::clamp(std::round(sum), 0.0, 255.0));
}

bool OtcheskovSGaussFilterVertSplitSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace otcheskov_s_gauss_filter_vert_split
