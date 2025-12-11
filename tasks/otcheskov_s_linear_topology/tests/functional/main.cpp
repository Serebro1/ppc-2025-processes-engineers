#include <gtest/gtest.h>
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

#include "otcheskov_s_linear_topology/common/include/common.hpp"
#include "otcheskov_s_linear_topology/mpi/include/ops_mpi.hpp"
#include "otcheskov_s_linear_topology/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace otcheskov_s_linear_topology {

class OtcheskovSLinearTopologyFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param).src) + "_" + std::to_string(std::get<0>(test_param).dest) +
           "_process_" + "test_" + std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.delivered;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

class OtcheskovSLinearTopologyFuncTestsValidation : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return FormatNumber(std::get<0>(test_param).src) + "_" + FormatNumber(std::get<0>(test_param).dest) + "_process_" +
           "test_" + std::to_string(std::get<1>(test_param));
  }

 protected:
  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.delivered;
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
  static std::string FormatNumber(int value) {
    std::stringstream ss;
    ss << value;
    std::string str = ss.str();
    if (value < 0) {
      str = "minus_" + str.substr(1, str.size());
    }
    return str;
  }

  InType input_data_;
  ppc::task::TaskPtr<InType, OutType> task_;
};

namespace {

TEST_P(OtcheskovSLinearTopologyFuncTests, LinearTopologyMpiFuncTests) {
  ExecuteTest(GetParam());
}

TEST_P(OtcheskovSLinearTopologyFuncTests, LinearTopologySeqFuncTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kMpiParam = {std::make_tuple(Message{0, 0, {1, 2, 3, 4, 5}, false}, 1),
                                           std::make_tuple(Message{0, 1, {1, 2, 3, 4, 5}, false}, 2),
                                           std::make_tuple(Message{1, 0, {1, 2, 3, 4, 5}, false}, 3),
                                           std::make_tuple(Message{1, 1, {1, 2, 3, 4, 5}, false}, 4)};

const std::array<TestType, 2> kSeqParam = {std::make_tuple(Message{0, 0, {1, 2, 3}, false}, 5),
                                           std::make_tuple(Message{0, 0, {42}, false}, 6)};

const auto kMpiTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<OtcheskovSLinearTopologyMPI, InType>(kMpiParam, PPC_SETTINGS_otcheskov_s_linear_topology));

const auto kSeqTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<OtcheskovSLinearTopologySEQ, InType>(kSeqParam, PPC_SETTINGS_otcheskov_s_linear_topology));

const auto kMpiGtestValues = ppc::util::ExpandToValues(kMpiTasksList);
const auto kSeqGtestValues = ppc::util::ExpandToValues(kSeqTasksList);

const auto kFuncTestName = OtcheskovSLinearTopologyFuncTests::PrintFuncTestName<OtcheskovSLinearTopologyFuncTests>;
INSTANTIATE_TEST_SUITE_P(LinearTopologyMpiFuncTests, OtcheskovSLinearTopologyFuncTests, kMpiGtestValues, kFuncTestName);
INSTANTIATE_TEST_SUITE_P(LinearTopologySeqFuncTests, OtcheskovSLinearTopologyFuncTests, kSeqGtestValues, kFuncTestName);

TEST_P(OtcheskovSLinearTopologyFuncTestsValidation, LinearTopologyTestsValidation) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 9> kValidationTestParam = {
    std::make_tuple(Message{-1, 0, {0}, false}, 1),  std::make_tuple(Message{0, -1, {0}, false}, 2),
    std::make_tuple(Message{-1, -1, {0}, false}, 3), std::make_tuple(Message{0, 0, {}, false}, 4),
    std::make_tuple(Message{-1, 0, {}, false}, 5),   std::make_tuple(Message{0, -1, {}, false}, 6),
    std::make_tuple(Message{-1, -2, {}, false}, 7),  std::make_tuple(Message{-1, -2, {}, false}, 8),
    std::make_tuple(Message{0, 0, {1}, true}, 9)};

const auto kValidationTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<OtcheskovSLinearTopologyMPI, InType>(
                       kValidationTestParam, PPC_SETTINGS_otcheskov_s_linear_topology),
                   ppc::util::AddFuncTask<OtcheskovSLinearTopologySEQ, InType>(
                       kValidationTestParam, PPC_SETTINGS_otcheskov_s_linear_topology));

const auto kValidationGtestValues = ppc::util::ExpandToValues(kValidationTestTasksList);

const auto kValidationFuncTestName =
    OtcheskovSLinearTopologyFuncTestsValidation::PrintFuncTestName<OtcheskovSLinearTopologyFuncTestsValidation>;

INSTANTIATE_TEST_SUITE_P(LinearTopologyTestsValidation, OtcheskovSLinearTopologyFuncTestsValidation,
                         kValidationGtestValues, kValidationFuncTestName);

}  // namespace
}  // namespace otcheskov_s_linear_topology
