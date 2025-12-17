#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

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
inline int mirrorCoord(int coord, int max) {
  if (coord < 0) {
    return -coord - 1;
  }
  if (coord >= max) {
    return 2 * max - coord - 1;
  }
  return coord;
}

InType ApplyGaussianFilter(const InType &input) {
  const auto &[data, height, width, channels] = input;
  OutType output;
  output.data.resize(data.size());
  output.height = height;
  output.width = width;
  output.channels = channels;

  if (height < 3 || width < 3) {
    output.data = data;
    return output;
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        double sum = 0.0;

        for (int ky = -1; ky <= 1; ++ky) {
          for (int kx = -1; kx <= 1; ++kx) {
            int ny = mirrorCoord(y + ky, height);
            int nx = mirrorCoord(x + kx, width);

            int idx = (ny * width + nx) * channels + c;
            sum += data[idx] * GAUSSIAN_KERNEL[ky + 1][kx + 1];
          }
        }

        int out_idx = (y * width + x) * channels + c;
        output.data[out_idx] = static_cast<uint8_t>(std::clamp(sum, 0.0, 255.0));
      }
    }
  }

  return output;
}

ImageData CreateGradientImage(int width, int height, int channels) {
  ImageData img;
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

// ImageData LoadRgbImageOrThrow(const std::string &task_id, const std::string &file_name) {
//   int width = -1;
//   int height = -1;
//   int channels_in_file = -1;

//   const std::string abs_path = ppc::util::GetAbsoluteTaskPath(task_id, file_name);
//   unsigned char *data = stbi_load(abs_path.c_str(), &width, &height, &channels_in_file, STBI_rgb);
//   if (data == nullptr) {
//     throw std::runtime_error("Failed to load image '" + abs_path + "': " + std::string(stbi_failure_reason()));
//   }

//   ImageData img;
//   img.width = width;
//   img.height = height;
//   img.channels = STBI_rgb;
//   const auto bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(img.channels);
//   img.data.assign(data, data + bytes);
//   stbi_image_free(data);
//   return img;
// }

// void save_image_as_jpeg(const std::string &filename, const ImageData &img, int quality = 90) {
//   int success = stbi_write_jpg(filename.c_str(), img.width, img.height, img.channels, img.data.data(), quality);
//   if (!success) {
//     throw std::runtime_error("Failed to save image as JPEG: " + filename);
//   }
// }

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
    input_img_ = CreateGradientImage(input_test_data.height, input_test_data.width, input_test_data.channels);
    expect_img_ = ApplyGaussianFilter(input_img_);
  }

  bool CheckTestOutputData(OutType &output_img) final {
    bool checked = false;
    if (!ppc::util::IsUnderMpirun()) {
      checked = expect_img_ == output_img;
    } else {
      int proc_rank{};
      MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
      if (proc_rank == 0) {
        checked = expect_img_ == output_img;
      } else {
        checked = true;
      }
    }
    return checked;
  }

  InType GetTestInputData() final {
    return input_img_;
  }

 private:
  InType input_img_;
  InType expect_img_;
};

// class OtcheskovSGaussFilterVertSplitRealTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType,
// TestType> {
//  public:
//   static std::string PrintTestParam(const TestType &test_param) {
//     return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
//   }

//  protected:
//   void SetUp() override {

//   }

//   bool CheckTestOutputData(OutType &output_data) final {
//     return false;
//   }

//   InType GetTestInputData() final {
//     return input_data_;
//   }

//  private:
//   InType input_data_;
// };

namespace {

const std::array<TestType, 6> kTestValidParam = {
    {{"empty_data", InType{.data = {}, .height = 3, .width = 3, .channels = 3}},
     {"image_2x2x1_not_valid", InType{.data = {10, 12, 14, 15}, .height = 2, .width = 2, .channels = 2}},
     {"image_3x3x1_wrong_size",
      InType{.data = {10, 12, 14, 15, 16, 17, 18, 19, 50}, .height = 4, .width = 4, .channels = 3}},
     {"image_3x3x1_wrong_height",
      InType{.data = {10, 12, 14, 15, 16, 17, 18, 19, 50}, .height = 0, .width = 3, .channels = 1}},
     {"image_3x3x1_wrong_Width",
      InType{.data = {10, 12, 14, 15, 16, 17, 18, 19, 50}, .height = 3, .width = 0, .channels = 1}},
     {"image_3x3x1_wrong_Channel",
      InType{.data = {10, 12, 14, 15, 16, 17, 18, 19, 50}, .height = 3, .width = 3, .channels = 0}}}};

const std::array<TestType, 4> kTestFuncParam = {
    {{"image_3x3x1", InType{.data = {}, .height = 3, .width = 3, .channels = 1}},
     {"image_3x3x3", InType{.data = {}, .height = 3, .width = 3, .channels = 3}},
     {"image_6x6x1", InType{.data = {}, .height = 6, .width = 6, .channels = 1}},
     {"image_21x21x3", InType{.data = {}, .height = 21, .width = 21, .channels = 3}}}};

const auto kTestValidTasksList = std::tuple_cat(ppc::util::AddFuncTask<OtcheskovSGaussFilterVertSplitMPI, InType>(
                                                    kTestValidParam, PPC_SETTINGS_otcheskov_s_gauss_filter_vert_split),
                                                ppc::util::AddFuncTask<OtcheskovSGaussFilterVertSplitSEQ, InType>(
                                                    kTestValidParam, PPC_SETTINGS_otcheskov_s_gauss_filter_vert_split));

const auto kTestFuncTasksList = std::tuple_cat(ppc::util::AddFuncTask<OtcheskovSGaussFilterVertSplitMPI, InType>(
                                                   kTestFuncParam, PPC_SETTINGS_otcheskov_s_gauss_filter_vert_split),
                                               ppc::util::AddFuncTask<OtcheskovSGaussFilterVertSplitSEQ, InType>(
                                                   kTestFuncParam, PPC_SETTINGS_otcheskov_s_gauss_filter_vert_split));

const auto kGtestValidValues = ppc::util::ExpandToValues(kTestValidTasksList);
const auto kGtestFuncValues = ppc::util::ExpandToValues(kTestFuncTasksList);

const auto kValidFuncTestName = OtcheskovSGaussFilterVertSplitValidationTestsProcesses::PrintFuncTestName<
    OtcheskovSGaussFilterVertSplitValidationTestsProcesses>;
const auto kFuncTestName = OtcheskovSGaussFilterVertSplitFuncTestsProcesses::PrintFuncTestName<
    OtcheskovSGaussFilterVertSplitFuncTestsProcesses>;

TEST_P(OtcheskovSGaussFilterVertSplitValidationTestsProcesses, Validation) {
  ExecuteTest(GetParam());
}

TEST_P(OtcheskovSGaussFilterVertSplitFuncTestsProcesses, Functional) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Validation, OtcheskovSGaussFilterVertSplitValidationTestsProcesses, kGtestValidValues,
                         kValidFuncTestName);

INSTANTIATE_TEST_SUITE_P(Functional, OtcheskovSGaussFilterVertSplitFuncTestsProcesses, kGtestFuncValues, kFuncTestName);
}  // namespace

}  // namespace otcheskov_s_gauss_filter_vert_split
