#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <ios>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "otcheskov_s_elem_vec_avg/common/include/common.hpp"
#include "otcheskov_s_elem_vec_avg/mpi/include/ops_mpi.hpp"
#include "otcheskov_s_elem_vec_avg/seq/include/ops_seq.hpp"
#include "task/include/task.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace otcheskov_s_elem_vec_avg {

class OtcheskovSElemVecAvgFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string filename = FormatFileName(std::get<0>(test_param));
    std::string avg_str = FormatAverage(std::get<1>(test_param));
    return filename + "_" + avg_str;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string filename = std::get<0>(params);
    expected_avg_ = std::get<1>(params);

    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_otcheskov_s_elem_vec_avg, filename);
    std::ifstream file(abs_path);

    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + abs_path);
    }

    int num{};
    while (file >> num) {
      input_data_.push_back(num);
    }
    file.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::fabs(expected_avg_ - output_data) < std::numeric_limits<double>::epsilon();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_avg_;

  static std::string FormatFileName(const std::string &filename) {
    size_t lastindex = filename.find_last_of('.');
    std::string name = filename;
    if (lastindex != std::string::npos) {
      name = filename.substr(0, lastindex);
    }

    std::string format_name = name;
    for (char &c : format_name) {
      if (std::isalnum(c) == 0 && c != '_') {
        c = '_';
      }
    }
    return format_name;
  }

  static std::string FormatAverage(double value) {
    std::string str = RemoveTrailingZeros(value);
    if (value < 0) {
      str = "minus_" + str.substr(1, str.size());
    }

    for (char &c : str) {
      if (c == '.') {
        c = 'p';
      }
    }
    return "num_" + str;
  }

  static std::string RemoveTrailingZeros(double value) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(10) << value;

    std::string str_value = ss.str();
    if (str_value.find('.') != std::string::npos) {
      str_value = str_value.substr(0, str_value.find_last_not_of('0') + 1);
      if (str_value.find('.') == str_value.size() - 1) {
        str_value = str_value.substr(0, str_value.size() - 1);
      }
    }
    return str_value;
  }
};

namespace {

TEST_P(OtcheskovSElemVecAvgFuncTests, VectorAverageFuncTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 7> kTestParam = {std::make_tuple("test_vec1.txt", 50.5),
                                            std::make_tuple("test_vec2.txt", 14.5),
                                            std::make_tuple("test_vec_one_elem.txt", 5.0),
                                            std::make_tuple("test_vec_fraction.txt", 4.0 / 3.0),
                                            std::make_tuple("test_vec_one_million_elems.txt", -2.60988),
                                            std::make_tuple("test_vec_alternating_elems.txt", 0.0),
                                            std::make_tuple("test_vec_zeros_elems.txt", 0.0)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<OtcheskovSElemVecAvgMPI, InType>(kTestParam, PPC_SETTINGS_otcheskov_s_elem_vec_avg),
    ppc::util::AddFuncTask<OtcheskovSElemVecAvgSEQ, InType>(kTestParam, PPC_SETTINGS_otcheskov_s_elem_vec_avg));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = OtcheskovSElemVecAvgFuncTests::PrintFuncTestName<OtcheskovSElemVecAvgFuncTests>;

INSTANTIATE_TEST_SUITE_P(VectorAverageFuncTests, OtcheskovSElemVecAvgFuncTests, kGtestValues, kFuncTestName);

// simple tests for sequential task
TEST(OtcheskovSElemVecAvgFuncTests, EmptyVectorSEQ) {
  InType vec;
  auto task_seq = ppc::task::TaskGetter<OtcheskovSElemVecAvgSEQ, InType>(vec);
  EXPECT_FALSE(task_seq->Validation());
  EXPECT_FALSE(task_seq->PreProcessing());
  EXPECT_FALSE(task_seq->Run());
  EXPECT_FALSE(task_seq->PostProcessing());
}

TEST(OtcheskovSElemVecAvgFuncTests, ChangedOutputBeforeRunSEQ) {
  InType vec = {1, 1, 1, 1, 1, 1};
  auto task_seq = ppc::task::TaskGetter<OtcheskovSElemVecAvgSEQ, InType>(vec);
  task_seq->GetOutput() = 1.0;
  EXPECT_FALSE(task_seq->Validation());
  EXPECT_FALSE(task_seq->PreProcessing());
  EXPECT_FALSE(task_seq->Run());
  task_seq->PostProcessing();
}

TEST(OtcheskovSElemVecAvgFuncTests, ChangedOutputAfterRunSEQ) {
  InType vec = {1, 1, 1, 1, 1, 1};
  auto task_seq = ppc::task::TaskGetter<OtcheskovSElemVecAvgSEQ, InType>(vec);

  EXPECT_TRUE(task_seq->Validation());
  EXPECT_TRUE(task_seq->PreProcessing());
  EXPECT_TRUE(task_seq->Run());
  task_seq->GetOutput() = NAN;
  // How to compare computed result in run with changed result?
  EXPECT_FALSE(task_seq->PostProcessing());
}

// simple tests for mpi task
TEST(OtcheskovSElemVecAvgFuncTests, EmptyVectorMPI) {
  InType vec;
  auto task_seq = ppc::task::TaskGetter<OtcheskovSElemVecAvgMPI, InType>(vec);
  EXPECT_FALSE(task_seq->Validation());
  EXPECT_FALSE(task_seq->PreProcessing());
  EXPECT_FALSE(task_seq->Run());
  EXPECT_FALSE(task_seq->PostProcessing());
}

TEST(OtcheskovSElemVecAvgFuncTests, ChangedOutputBeforeRunMPI) {
  InType vec = {1, 1, 1, 1, 1, 1};
  auto task_seq = ppc::task::TaskGetter<OtcheskovSElemVecAvgMPI, InType>(vec);
  task_seq->GetOutput() = 1.0;
  EXPECT_FALSE(task_seq->Validation());
  EXPECT_FALSE(task_seq->PreProcessing());
  EXPECT_FALSE(task_seq->Run());
  task_seq->PostProcessing();
}

TEST(OtcheskovSElemVecAvgFuncTests, ChangedOutputAfterRunMPI) {
  InType vec = {1, 1, 1, 1, 1, 1};
  auto task_seq = ppc::task::TaskGetter<OtcheskovSElemVecAvgMPI, InType>(vec);
  EXPECT_TRUE(task_seq->Validation());
  EXPECT_TRUE(task_seq->PreProcessing());
  EXPECT_TRUE(task_seq->Run());
  task_seq->GetOutput() = NAN;
  // How to compare computed result in run with changed result?
  EXPECT_FALSE(task_seq->PostProcessing());
}

}  // namespace

}  // namespace otcheskov_s_elem_vec_avg
