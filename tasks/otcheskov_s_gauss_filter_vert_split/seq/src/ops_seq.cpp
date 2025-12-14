#include "otcheskov_s_gauss_filter_vert_split/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstdint>
#include <numeric>

#include "otcheskov_s_gauss_filter_vert_split/common/include/common.hpp"

namespace otcheskov_s_gauss_filter_vert_split {

otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitSEQ::OtcheskovSGaussFilterVertSplitSEQ(
    const InType &in) {
  GetInput() = in;

  const auto &input = GetInput();
  GetOutput().height = input.height;
  GetOutput().width = input.width;
  GetOutput().channels = input.channels;
  GetOutput().data.resize(input.data.size());
}

int otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitSEQ::GetIndex(int row, int col, int channel) {
  return static_cast<std::size_t>(row * GetInput().width + col) * GetInput().channels + channel;
}

bool otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.height < 3 || input.width < 3) {
    return false;
  }

  size_t expected_size = input.height * input.width * input.channels;
  if (input.data.size() != expected_size) {
    return false;
  }

  return true;
}

bool otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitSEQ::PreProcessingImpl() {
  return true;
}

bool otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();
  int channels = input.channels;

  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < input.height; ++i) {
      output.data[GetIndex(i, 0, c)] = input.data[GetIndex(i, 0, c)];
      output.data[GetIndex(i, input.width - 1, c)] = input.data[GetIndex(i, input.width - 1, c)];
    }
    for (int j = 0; j < input.width; ++j) {
      output.data[GetIndex(0, j, c)] = input.data[GetIndex(0, j, c)];
      output.data[GetIndex(input.height - 1, j, c)] = input.data[GetIndex(input.height - 1, j, c)];
    }
    for (int i = 1; i < input.height - 1; ++i) {
      for (int j = 1; j < input.width - 1; ++j) {
        double sum = 0.0;

        // Свёртка с ядром Гаусса для текущего канала
        for (int ki = -1; ki <= 1; ++ki) {
          for (int kj = -1; kj <= 1; ++kj) {
            int idx = GetIndex(i + ki, j + kj, c);
            sum += input.data[idx] * GAUSSIAN_KERNEL[ki + 1][kj + 1];
          }
        }

        output.data[GetIndex(i, j, c)] = static_cast<uint8_t>(std::clamp(sum, 0.0, 255.0));
        ;
      }
    }
  }
  return true;
}

bool otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitSEQ::PostProcessingImpl() {
  return false;
}

}  // namespace otcheskov_s_gauss_filter_vert_split
