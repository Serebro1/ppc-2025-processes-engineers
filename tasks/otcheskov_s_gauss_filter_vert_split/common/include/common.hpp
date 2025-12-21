#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace otcheskov_s_gauss_filter_vert_split {

struct ImageMetadata {
  size_t height{};
  size_t width{};
  size_t channels{};

  bool operator==(const ImageMetadata &other) const {
    return height == other.height && width == other.width && channels == other.channels;
  }
  bool operator!=(const ImageMetadata &other) const {
    return !(*this == other);
  }
};

using ImageData = std::vector<uint8_t>;
using Image = std::pair<ImageMetadata, ImageData>;

using InType = Image;
using OutType = Image;
using TestType = std::tuple<std::string, InType>;
using BaseTask = ppc::task::Task<InType, OutType>;

constexpr std::array<std::array<double, 3>, 3> kGaussianKernel = {
    {{1.0 / 16, 2.0 / 16, 1.0 / 16}, {2.0 / 16, 4.0 / 16, 2.0 / 16}, {1.0 / 16, 2.0 / 16, 1.0 / 16}}};

}  // namespace otcheskov_s_gauss_filter_vert_split
