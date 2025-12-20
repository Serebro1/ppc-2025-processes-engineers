#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "otcheskov_s_gauss_filter_vert_split/common/include/common.hpp"
#include "otcheskov_s_gauss_filter_vert_split/mpi/include/ops_mpi.hpp"
#include "otcheskov_s_gauss_filter_vert_split/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace otcheskov_s_gauss_filter_vert_split {
namespace {

InType ApplyGaussianFilter(const InType &input) {
  const auto &[data, height, width, channels] = input;
  OutType output{.data = std::vector<uint8_t>(data.size()), .height = height, .width = width, .channels = channels};

  auto mirrorCoord = [](int i, int size) {
    if (i < 0) {
      return -i - 1;
    }
    if (i >= size) {
      return 2 * size - i - 1;
    }
    return i;
  };

  for (int y = 0; y < input.height; ++y) {
    for (int x = 0; x < input.width; ++x) {
      for (int c = 0; c < input.channels; ++c) {
        double sum = 0.0;
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            int srcY = mirrorCoord(y + dy, input.height);
            int srcX = mirrorCoord(x + dx, input.width);
            int idx = (srcY * input.width + srcX) * input.channels + c;
            sum += input.data[idx] * kGaussianKernel[dy + 1][dx + 1];
          }
        }
        int out_idx = (y * input.width + x) * input.channels + c;
        output.data[out_idx] = static_cast<uint8_t>(std::clamp(std::round(sum), 0.0, 255.0));
      }
    }
  }
  return output;
}

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

InType LoadRgbImage(const std::string &img_path) {
  int width = -1;
  int height = -1;
  int channels_in_file = -1;

  unsigned char *data = stbi_load(img_path.c_str(), &width, &height, &channels_in_file, STBI_rgb);
  if (data == nullptr) {
    throw std::runtime_error("Failed to load image '" + img_path + "': " + std::string(stbi_failure_reason()));
  }

  InType img;
  img.width = width;
  img.height = height;
  img.channels = STBI_rgb;
  const auto bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(img.channels);
  img.data.assign(data, data + bytes);
  stbi_image_free(data);
  return img;
}

}  // namespace

class OtcheskovSGaussFilterVertSplitValidationTestsProcesses
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  void ExecuteTest(::ppc::util::FuncTestParam<InType, OutType, TestType> test_param) {
    const std::string &test_name =
        std::get<static_cast<std::size_t>(::ppc::util::GTestParamIndex::kNameTest)>(test_param);

    ValidateTestName(test_name);

    const auto test_env_scope = ppc::util::test::MakePerTestEnvForCurrentGTest(test_name);

    if (IsTestDisabled(test_name)) {
      GTEST_SKIP();
    }

    if (ShouldSkipNonMpiTask(test_name)) {
      std::cerr << "kALL and kMPI tasks are not under mpirun\n";
      GTEST_SKIP();
    }

    task_ =
        std::get<static_cast<std::size_t>(::ppc::util::GTestParamIndex::kTaskGetter)>(test_param)(GetTestInputData());
    const TestType &params = std::get<static_cast<std::size_t>(::ppc::util::GTestParamIndex::kTestParams)>(test_param);
    const std::string param_name = std::get<0>(params);
    task_->GetInput() = std::get<1>(params);
    ExecuteTaskPipeline();
  }

  void ExecuteTaskPipeline() {
    EXPECT_FALSE(task_->Validation());
    task_->PreProcessing();
    task_->Run();
    task_->PostProcessing();
  }

 private:
  InType input_data_;
  ppc::task::TaskPtr<InType, OutType> task_;
};

class OtcheskovSGaussFilterVertSplitFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
  void SetUp() override {
    const TestType &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const InType &input_test_data = std::get<1>(params);
    input_img_ = CreateGradientImage(input_test_data.width, input_test_data.height, input_test_data.channels);
    expect_img_ = ApplyGaussianFilter(input_img_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (!ppc::util::IsUnderMpirun()) {
      return expect_img_ == output_data;
    }

    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    if (proc_rank == 0) {
      return expect_img_ == output_data;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_img_;
  }

 private:
  InType input_img_;
  InType expect_img_;
};

class OtcheskovSGaussFilterVertSplitRealTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string filename = std::get<0>(test_param);

    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos != std::string::npos) {
      filename = filename.substr(0, dot_pos);
    }

    return filename;
  }

 protected:
  void SetUp() override {
    try {
      const TestType &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
      const std::string &filename = std::get<0>(params);
      std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_otcheskov_s_gauss_filter_vert_split, filename);
      input_img_ = LoadRgbImage(abs_path);
      expect_img_ = ApplyGaussianFilter(LoadRgbImage(abs_path));
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n';
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (!ppc::util::IsUnderMpirun()) {
      return expect_img_ == output_data;
    }

    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    if (proc_rank == 0) {
      return expect_img_ == output_data;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_img_;
  }

 private:
  InType input_img_;
  InType expect_img_;
};

namespace {

const std::array<TestType, 6> kTestValidParam = {
    {{"empty_data", InType{.data = {}, .height = 3, .width = 3, .channels = 3}},
     {"image_2x2x1_not_valid", InType{.data = {10, 12, 14, 15}, .height = 2, .width = 2, .channels = 2}},
     {"image_3x3x1_wrong_size",
      InType{.data = {10, 12, 14, 15, 16, 17, 18, 19, 50}, .height = 4, .width = 4, .channels = 3}},
     {"image_3x3x1_wrong_height",
      InType{.data = {10, 12, 14, 15, 16, 17, 18, 19, 50}, .height = 0, .width = 3, .channels = 1}},
     {"image_3x3x1_wrong_width",
      InType{.data = {10, 12, 14, 15, 16, 17, 18, 19, 50}, .height = 3, .width = 0, .channels = 1}},
     {"image_3x3x1_wrong_channel",
      InType{.data = {10, 12, 14, 15, 16, 17, 18, 19, 50}, .height = 3, .width = 3, .channels = 0}}}};

const std::array<TestType, 7> kTestFuncParam = {
    {{"image_3x3x1", InType{.data = {}, .height = 3, .width = 3, .channels = 1}},
     {"image_3x3x3", InType{.data = {}, .height = 3, .width = 3, .channels = 3}},
     {"image_4x4x1", InType{.data = {}, .height = 4, .width = 4, .channels = 1}},
     {"image_10x20x3", InType{.data = {}, .height = 10, .width = 20, .channels = 3}},
     {"border_test_9x9", {.data = {}, .height = 9, .width = 9, .channels = 1}},
     {"border_test_10x10", {.data = {}, .height = 10, .width = 10, .channels = 1}},
     {"sharp_vertical_lines_15x15", {.data = {}, .height = 15, .width = 15, .channels = 3}}}};

const std::array<TestType, 2> kTestRealParam = {
    {{"chess.jpg", InType{.data = {}, .height = {}, .width = {}, .channels = {}}},
     {"gradient.jpg", InType{.data = {}, .height = {}, .width = {}, .channels = {}}}}};

const auto kTestValidTasksList = std::tuple_cat(ppc::util::AddFuncTask<OtcheskovSGaussFilterVertSplitMPI, InType>(
                                                    kTestValidParam, PPC_SETTINGS_otcheskov_s_gauss_filter_vert_split),
                                                ppc::util::AddFuncTask<OtcheskovSGaussFilterVertSplitSEQ, InType>(
                                                    kTestValidParam, PPC_SETTINGS_otcheskov_s_gauss_filter_vert_split));

const auto kTestFuncTasksList = std::tuple_cat(ppc::util::AddFuncTask<OtcheskovSGaussFilterVertSplitMPI, InType>(
                                                   kTestFuncParam, PPC_SETTINGS_otcheskov_s_gauss_filter_vert_split),
                                               ppc::util::AddFuncTask<OtcheskovSGaussFilterVertSplitSEQ, InType>(
                                                   kTestFuncParam, PPC_SETTINGS_otcheskov_s_gauss_filter_vert_split));

const auto kTestRealTasksList = std::tuple_cat(ppc::util::AddFuncTask<OtcheskovSGaussFilterVertSplitMPI, InType>(
                                                   kTestRealParam, PPC_SETTINGS_otcheskov_s_gauss_filter_vert_split),
                                               ppc::util::AddFuncTask<OtcheskovSGaussFilterVertSplitSEQ, InType>(
                                                   kTestRealParam, PPC_SETTINGS_otcheskov_s_gauss_filter_vert_split));

const auto kGtestValidValues = ppc::util::ExpandToValues(kTestValidTasksList);
const auto kGtestFuncValues = ppc::util::ExpandToValues(kTestFuncTasksList);
const auto kGtestRealValues = ppc::util::ExpandToValues(kTestRealTasksList);

const auto kValidFuncTestName = OtcheskovSGaussFilterVertSplitValidationTestsProcesses::PrintFuncTestName<
    OtcheskovSGaussFilterVertSplitValidationTestsProcesses>;

const auto kFuncTestName = OtcheskovSGaussFilterVertSplitFuncTestsProcesses::PrintFuncTestName<
    OtcheskovSGaussFilterVertSplitFuncTestsProcesses>;

const auto kRealTestName = OtcheskovSGaussFilterVertSplitRealTestsProcesses::PrintFuncTestName<
    OtcheskovSGaussFilterVertSplitRealTestsProcesses>;

TEST_P(OtcheskovSGaussFilterVertSplitValidationTestsProcesses, Validation) {
  ExecuteTest(GetParam());
}

TEST_P(OtcheskovSGaussFilterVertSplitFuncTestsProcesses, Functional) {
  ExecuteTest(GetParam());
}

TEST_P(OtcheskovSGaussFilterVertSplitRealTestsProcesses, RealImages) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Validation, OtcheskovSGaussFilterVertSplitValidationTestsProcesses, kGtestValidValues,
                         kValidFuncTestName);

INSTANTIATE_TEST_SUITE_P(Functional, OtcheskovSGaussFilterVertSplitFuncTestsProcesses, kGtestFuncValues, kFuncTestName);

INSTANTIATE_TEST_SUITE_P(RealImages, OtcheskovSGaussFilterVertSplitRealTestsProcesses, kGtestRealValues, kRealTestName);

}  // namespace

}  // namespace otcheskov_s_gauss_filter_vert_split
