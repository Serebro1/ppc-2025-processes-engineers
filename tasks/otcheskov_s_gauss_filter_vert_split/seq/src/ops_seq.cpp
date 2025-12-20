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

int otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitSEQ::GetIndex(int row, int col, int channel) {
  return (((row * GetInput().width) + col) * GetInput().channels) + channel;
}

bool otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitSEQ::ValidationImpl() {
  const auto &input = GetInput();
  bool is_valid = false;
  is_valid = !input.data.empty() && (input.height >= 3 && input.width >= 3 && input.channels > 0) &&
             (input.data.size() == static_cast<std::size_t>(input.height) * static_cast<size_t>(input.width) *
                                       static_cast<size_t>(input.channels));

  return is_valid;
}

bool otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitSEQ::PreProcessingImpl() {
  return true;
}

bool otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitSEQ::RunImpl() {
  const auto &input = GetInput();
  bool is_valid = !input.data.empty() && (input.height >= 3 && input.width >= 3 && input.channels > 0) &&
                  (input.data.size() == static_cast<std::size_t>(input.height) * static_cast<size_t>(input.width) *
                                            static_cast<size_t>(input.channels));
  if (!is_valid) {
    return false;
  }
  auto &output = GetOutput();
  output.height = input.height;
  output.width = input.width;
  output.channels = input.channels;
  output.data.resize(input.data.size());

  for (int y = 0; y < input.height; ++y) {
    for (int x = 0; x < input.width; ++x) {
      for (int ch = 0; ch < input.channels; ++ch) {
        double sum = 0.0;

        for (int ky = -1; ky <= 1; ++ky) {
          int yk = y + ky;
          for (int kx = -1; kx <= 1; ++kx) {
            int xk = x + kx;

            if (yk < 0) {
              yk = -yk - 1;
            } else if (yk >= input.height) {
              yk = (2 * input.height) - yk - 1;
            }

            if (xk < 0) {
              xk = -xk - 1;
            } else if (xk >= input.width) {
              xk = (2 * input.width) - xk - 1;
            }

            double weight = kGaussianKernel.at(ky + 1).at(kx + 1);

            int idx = GetIndex(yk, xk, ch);
            sum += weight * input.data[idx];
          }
        }

        int out_idx = GetIndex(y, x, ch);
        output.data[out_idx] = static_cast<uint8_t>(std::clamp(std::round(sum), 0.0, 255.0));
      }
    }
  }
  return true;
}

bool otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace otcheskov_s_gauss_filter_vert_split
