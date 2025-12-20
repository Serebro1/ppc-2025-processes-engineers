#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>

#include "otcheskov_s_gauss_filter_vert_split/common/include/common.hpp"
#include "otcheskov_s_gauss_filter_vert_split/mpi/include/ops_mpi.hpp"
#include "otcheskov_s_gauss_filter_vert_split/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace otcheskov_s_gauss_filter_vert_split {
namespace {
InType CreateGradientImage(int width, int height, int channels) {
  InType img;
  img.width = width;
  img.height = height;
  img.channels = channels;

  const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(channels);
  img.data.resize(pixel_count);

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      for (int ch = 0; ch < channels; ++ch) {
        const size_t idx = (static_cast<size_t>(row) * static_cast<size_t>(width) * static_cast<size_t>(channels)) +
                           (static_cast<size_t>(col) * static_cast<size_t>(channels)) + static_cast<size_t>(ch);
        img.data[idx] = static_cast<uint8_t>((col * 2 + row + ch * 50) % 256);
      }
    }
  }

  return img;
}

}  // namespace

class OtcheskovSGaussFilterVertSplitPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kMatrixSize = 1000;
  InType input_img_;

  void SetUp() override {
    input_img_ = CreateGradientImage(kMatrixSize, kMatrixSize, 1);
  }

  bool CheckTestOutputData(OutType &output_img) final {
    bool is_checked = false;
    if (!ppc::util::IsUnderMpirun()) {
      is_checked = output_img.channels == input_img_.channels && output_img.height == input_img_.height &&
                   output_img.width == output_img.width;
    } else {
      int proc_rank{};
      MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
      if (proc_rank == 0) {
        is_checked = output_img.channels == input_img_.channels && output_img.height == input_img_.height &&
                     output_img.width == output_img.width;
      } else {
        is_checked = true;
      }
    }
    return is_checked;
  }

  InType GetTestInputData() final {
    return input_img_;
  }
};

TEST_P(OtcheskovSGaussFilterVertSplitPerfTests, RunPerfTests) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, OtcheskovSGaussFilterVertSplitMPI, OtcheskovSGaussFilterVertSplitSEQ>(
        PPC_SETTINGS_otcheskov_s_gauss_filter_vert_split);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = OtcheskovSGaussFilterVertSplitPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunPerfTests, OtcheskovSGaussFilterVertSplitPerfTests, kGtestValues, kPerfTestName);

}  // namespace otcheskov_s_gauss_filter_vert_split
