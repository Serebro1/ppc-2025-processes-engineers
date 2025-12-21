#include "otcheskov_s_gauss_filter_vert_split/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "otcheskov_s_gauss_filter_vert_split/common/include/common.hpp"

namespace otcheskov_s_gauss_filter_vert_split {

otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitSEQ::OtcheskovSGaussFilterVertSplitSEQ(
    const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

size_t otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitSEQ::GetIndex(size_t row, size_t col,
                                                                                        size_t channel) {
  const auto &metadata = GetInput().first;
  return (((row * metadata.width) + col) * metadata.channels) + channel;
}

bool otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitSEQ::ValidationImpl() {
  const auto &[metadata, data] = GetInput();

  is_valid_ = !data.empty() && (metadata.height >= 3 && metadata.width >= 3 && metadata.channels > 0) &&
              (data.size() == metadata.height * metadata.width * metadata.channels);

  return is_valid_;
}

bool otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitSEQ::PreProcessingImpl() {
  return true;
}

bool otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitSEQ::RunImpl() {
  const auto &[in_meta, in_data] = GetInput();
  if (!is_valid_) {
    return false;
  }
  auto &[out_meta, out_data] = GetOutput();
  out_meta = in_meta;
  out_data.resize(in_data.size());

  const auto &[height, width, channels] = in_meta;

  const auto process_pixel = [&](size_t row, size_t col, size_t ch) -> uint8_t {
    double sum = 0.0;

    for (int dy = -1; dy <= 1; ++dy) {
      size_t src_y = MirrorCoord(row, dy, height);
      for (int dx = -1; dx <= 1; ++dx) {
        size_t src_x = MirrorCoord(col, dx, width);
        double weight = kGaussianKernel.at(dy + 1).at(dx + 1);
        size_t src_idx = GetIndex(src_y, src_x, ch);
        sum += weight * in_data[src_idx];
      }
    }

    return static_cast<uint8_t>(std::clamp(std::round(sum), 0.0, 255.0));
  };

  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      for (size_t ch = 0; ch < channels; ++ch) {
        size_t out_idx = GetIndex(row, col, ch);
        out_data[out_idx] = process_pixel(row, col, ch);
      }
    }
  }
  return true;
}

size_t OtcheskovSGaussFilterVertSplitSEQ::MirrorCoord(size_t current, int off, size_t size) {
  int64_t pos = static_cast<int64_t>(current) + off;
  if (pos < 0) {
    return static_cast<size_t>(-pos - 1);
  }
  if (std::cmp_greater_equal(static_cast<size_t>(pos), size)) {
    return (2 * size) - static_cast<size_t>(pos) - 1;
  }
  return static_cast<size_t>(pos);
}

bool otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace otcheskov_s_gauss_filter_vert_split
