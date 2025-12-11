#include <gtest/gtest.h>

#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "gaivoronskiy_m_average_vector_sum/common/include/common.hpp"
#include "gaivoronskiy_m_average_vector_sum/mpi/include/ops_mpi.hpp"
#include "gaivoronskiy_m_average_vector_sum/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace gaivoronskiy_m_average_vector_sum {

namespace {

std::vector<double> LoadVectorFromFile(const std::string &file_name) {
  std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_gaivoronskiy_m_average_vector_sum, file_name);
  std::ifstream file(abs_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + abs_path);
  }
  std::vector<double> data;
  double value = 0.0;
  while (file >> value) {
    data.push_back(value);
  }
  return data;
}

std::string StripExtension(const std::string &file_name) {
  const auto pos = file_name.find_last_of('.');
  if (pos == std::string::npos) {
    return file_name;
  }
  return file_name.substr(0, pos);
}

std::string SanitizeToken(std::string token) {
  for (char &ch : token) {
    if (std::isalnum(static_cast<unsigned char>(ch)) == 0) {
      ch = '_';
    }
  }
  return token;
}

}  // namespace

class AverageVectorSumFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return SanitizeToken(StripExtension(std::get<0>(test_param)));
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = LoadVectorFromFile(std::get<0>(params));
    expected_average_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double k_eps = 1e-9;
    return std::fabs(output_data - expected_average_) <= k_eps;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_average_ = 0.0;
};

namespace {

const std::array<TestType, 5> kTestCases = {{
    {"test_vec_small.txt", 5.5},
    {"test_vec_mixed.txt", 2.0},
    {"test_vec_single.txt", 42.5},
    {"test_vec_progression.txt", 0.0},
    {"test_vec_fraction.txt", 2.5},
}};

TEST_P(AverageVectorSumFuncTests, ComputesAverageCorrectly) {
  ExecuteTest(GetParam());
}

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<GaivoronskiyMAverageVecSumMPI, InType>(
                                               kTestCases, PPC_SETTINGS_gaivoronskiy_m_average_vector_sum),
                                           ppc::util::AddFuncTask<GaivoronskiyMAverageVecSumSEQ, InType>(
                                               kTestCases, PPC_SETTINGS_gaivoronskiy_m_average_vector_sum));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = AverageVectorSumFuncTests::PrintFuncTestName<AverageVectorSumFuncTests>;

INSTANTIATE_TEST_SUITE_P(AverageCases, AverageVectorSumFuncTests, kGtestValues, kPerfTestName);

class AverageVectorSumValidationTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return SanitizeToken(StripExtension(std::get<0>(test_param)));
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = LoadVectorFromFile(std::get<0>(params));
  }

  bool CheckTestOutputData(OutType & /*output_data*/) final {
    return true;
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

const std::array<TestType, 1> kValidationCases = {{
    {"test_empty_vec.txt", 0.0},
}};

TEST_P(AverageVectorSumValidationTests, HandlesInvalidInput) {
  ExecuteTest(GetParam());
}

const auto kValidationTasks = std::tuple_cat(ppc::util::AddFuncTask<GaivoronskiyMAverageVecSumMPI, InType>(
                                                 kValidationCases, PPC_SETTINGS_gaivoronskiy_m_average_vector_sum),
                                             ppc::util::AddFuncTask<GaivoronskiyMAverageVecSumSEQ, InType>(
                                                 kValidationCases, PPC_SETTINGS_gaivoronskiy_m_average_vector_sum));

const auto kValidationValues = ppc::util::ExpandToValues(kValidationTasks);

const auto kValidationNames = AverageVectorSumValidationTests::PrintFuncTestName<AverageVectorSumValidationTests>;

INSTANTIATE_TEST_SUITE_P(AverageValidationCases, AverageVectorSumValidationTests, kValidationValues, kValidationNames);

}  // namespace

}  // namespace gaivoronskiy_m_average_vector_sum
