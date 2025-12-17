#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace otcheskov_s_gauss_filter_vert_split {

struct ImageData {
  std::vector<uint8_t> data;
  int height{};
  int width{};
  int channels{};

  bool operator==(const ImageData &other) const {
    return height == other.height && width == other.width && channels == other.channels && data == other.data;
  }
  bool operator!=(const ImageData &other) const {
    return !(*this == other);
  }
};

using InType = ImageData;
using OutType = ImageData;
using TestType = std::tuple<std::string, InType>;
using BaseTask = ppc::task::Task<InType, OutType>;

constexpr double GAUSSIAN_KERNEL[3][3] = {
    {1.0 / 16, 2.0 / 16, 1.0 / 16}, {2.0 / 16, 4.0 / 16, 2.0 / 16}, {1.0 / 16, 2.0 / 16, 1.0 / 16}};

}  // namespace otcheskov_s_gauss_filter_vert_split
