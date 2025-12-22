#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
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
  const auto &[metadata, image_data] = input;
  const auto &[height, width, channels] = metadata;
  const auto &data = image_data;

  ImageMetadata out_metadata = metadata;
  ImageData out_data = std::vector<uint8_t>(data.size());

  auto mirror_coord = [](size_t curr, int off, size_t size) {
    int64_t pos = static_cast<int64_t>(curr) + off;
    if (pos < 0) {
      return static_cast<size_t>(-pos - 1);
    }
    if (std::cmp_greater_equal(static_cast<size_t>(pos), size)) {
      return static_cast<size_t>((2 * size) - pos - 1);
    }
    return static_cast<size_t>(pos);
  };

  const size_t row_stride = width * channels;

  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      for (size_t ch = 0; ch < channels; ++ch) {
        double sum = 0.0;
        for (int dy = -1; dy <= 1; ++dy) {
          size_t src_y = mirror_coord(row, dy, height);
          for (int dx = -1; dx <= 1; ++dx) {
            size_t src_x = mirror_coord(col, dx, width);
            size_t src_idx = (src_y * row_stride) + (src_x * channels) + ch;
            sum += data[src_idx] * kGaussianKernel.at(dy + 1).at(dx + 1);
          }
        }
        size_t out_idx = (row * row_stride) + (col * channels) + ch;
        out_data[out_idx] = static_cast<uint8_t>(std::clamp(std::round(sum), 0.0, 255.0));
      }
    }
  }
  return {out_metadata, out_data};
}

InType CreateGradientImage(ImageMetadata img_metadata) {
  ImageData img_data;
  const auto &[width, height, channels] = img_metadata;
  img_data.resize(width * height * channels);

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

InType LoadRgbImage(const std::string &img_path) {
  int width = -1;
  int height = -1;
  int channels_in_file = -1;

  unsigned char *data = stbi_load(img_path.c_str(), &width, &height, &channels_in_file, STBI_rgb);
  if (data == nullptr) {
    throw std::runtime_error("Failed to load image '" + img_path + "': " + std::string(stbi_failure_reason()));
  }

  ImageMetadata img_metadata;
  ImageData img_data;
  img_metadata.width = static_cast<size_t>(width);
  img_metadata.height = static_cast<size_t>(height);
  img_metadata.channels = STBI_rgb;
  const auto bytes = img_metadata.width * img_metadata.height * img_metadata.channels;
  img_data.assign(data, data + bytes);
  stbi_image_free(data);
  return {img_metadata, img_data};
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
    return output_data.second.empty();
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
    input_img_ = CreateGradientImage(input_test_data.first);
    expect_img_ = ApplyGaussianFilter(input_img_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (!ppc::util::IsUnderMpirun()) {
      return expect_img_ == output_data;
    }

    int proc_rank{};
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

    int proc_rank{};
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

const std::array<TestType, 5> kTestValidParam = {
    {{"empty_data", {ImageMetadata{.height = 3, .width = 3, .channels = 3}, ImageData{}}},
     {"image_3x3x1_wrong_size",
      {ImageMetadata{.height = 4, .width = 4, .channels = 3}, ImageData{10, 12, 14, 15, 16, 17, 18, 19, 50}}},
     {"image_3x3x1_wrong_height",
      {ImageMetadata{.height = 0, .width = 3, .channels = 1}, ImageData{10, 12, 14, 15, 16, 17, 18, 19, 50}}},
     {"image_3x3x1_wrong_width",
      {ImageMetadata{.height = 3, .width = 0, .channels = 1}, ImageData{10, 12, 14, 15, 16, 17, 18, 19, 50}}},
     {"image_3x3x1_wrong_channel",
      {ImageMetadata{.height = 3, .width = 3, .channels = 0}, ImageData{10, 12, 14, 15, 16, 17, 18, 19, 50}}}}};

const std::array<TestType, 8> kTestFuncParam = {
    {{"image_15x1x1", {ImageMetadata{.height = 15, .width = 1, .channels = 1}, ImageData{}}},
     {"image_3x3x1", {ImageMetadata{.height = 3, .width = 3, .channels = 1}, ImageData{}}},
     {"image_3x3x3", {ImageMetadata{.height = 3, .width = 3, .channels = 3}, ImageData{}}},
     {"image_4x4x1", {ImageMetadata{.height = 4, .width = 4, .channels = 1}, ImageData{}}},
     {"image_10x20x3", {ImageMetadata{.height = 10, .width = 20, .channels = 3}, ImageData{}}},
     {"border_test_9x9", {ImageMetadata{.height = 9, .width = 9, .channels = 1}, ImageData{}}},
     {"border_test_10x10", {ImageMetadata{.height = 10, .width = 10, .channels = 1}, ImageData{}}},
     {"sharp_vertical_lines_15x15", {ImageMetadata{.height = 15, .width = 15, .channels = 3}, ImageData{}}}}};

const std::array<TestType, 2> kTestRealParam = {
    {{"chess.jpg", {ImageMetadata{}, ImageData{}}}, {"gradient.jpg", {ImageMetadata{}, ImageData{}}}}};

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

TEST_P(OtcheskovSGaussFilterVertSplitValidationTestsProcesses, GaussFilterVertSplitValidation) {
  ExecuteTest(GetParam());
}

TEST_P(OtcheskovSGaussFilterVertSplitFuncTestsProcesses, GaussFilterVertSplitFunc) {
  ExecuteTest(GetParam());
}

TEST_P(OtcheskovSGaussFilterVertSplitRealTestsProcesses, GaussFilterVertSplitReal) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(GaussFilterVertSplitValidation, OtcheskovSGaussFilterVertSplitValidationTestsProcesses,
                         kGtestValidValues, kValidFuncTestName);

INSTANTIATE_TEST_SUITE_P(GaussFilterVertSplitFunc, OtcheskovSGaussFilterVertSplitFuncTestsProcesses, kGtestFuncValues,
                         kFuncTestName);

INSTANTIATE_TEST_SUITE_P(GaussFilterVertSplitReal, OtcheskovSGaussFilterVertSplitRealTestsProcesses, kGtestRealValues,
                         kRealTestName);

}  // namespace

}  // namespace otcheskov_s_gauss_filter_vert_split
