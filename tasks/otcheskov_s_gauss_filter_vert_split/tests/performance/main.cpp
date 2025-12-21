#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "otcheskov_s_gauss_filter_vert_split/common/include/common.hpp"
#include "otcheskov_s_gauss_filter_vert_split/mpi/include/ops_mpi.hpp"
#include "otcheskov_s_gauss_filter_vert_split/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace otcheskov_s_gauss_filter_vert_split {
namespace {
InType CreateGradientImage(ImageMetadata img_metadata) {
  ImageData img_data;
  const auto &[height, width, channels] = img_metadata;
  const size_t pixel_count = width * height * channels;
  img_data.resize(pixel_count);

  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      for (size_t ch = 0; ch < channels; ++ch) {
        const size_t idx = (row * width * channels) + (col * channels) + ch;
        img_data[idx] = static_cast<uint8_t>((col * 2 + row + ch * 50) % 256);
      }
    }
  }

  return {img_metadata, img_data};
}

}  // namespace

class OtcheskovSGaussFilterVertSplitPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr size_t kMatrixSize = 10000;
  InType input_img_;

  void SetUp() override {
    input_img_ = CreateGradientImage(ImageMetadata{.height = kMatrixSize, .width = kMatrixSize, .channels = 3});
  }

  bool CheckTestOutputData(OutType &output_img) final {
    bool is_checked = false;
    if (!ppc::util::IsUnderMpirun()) {
      is_checked = output_img.first == input_img_.first;
    } else {
      int proc_rank{};
      MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
      if (proc_rank == 0) {
        is_checked = output_img.first == input_img_.first;
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
