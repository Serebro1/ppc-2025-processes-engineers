#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>

#include "otcheskov_s_linear_topology/common/include/common.hpp"
#include "otcheskov_s_linear_topology/mpi/include/ops_mpi.hpp"
#include "otcheskov_s_linear_topology/seq/include/ops_seq.hpp"
#include "task/include/task.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace otcheskov_s_linear_topology {

class OtcheskovSLinearTopologyFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param).first.src) + "_" +
           std::to_string(std::get<0>(test_param).first.dest) + "_process_" + "test_" +
           std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_msg = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_msg) final {
    const auto &[in_header, in_data] = input_msg;
    const auto &[out_header, out_data] = output_msg;

    bool is_valid = false;
    if (!ppc::util::IsUnderMpirun()) {
      is_valid = out_header.delivered != 0;
    } else {
      int proc_rank{};
      MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
      if (proc_rank == in_header.src || proc_rank == in_header.dest) {
        is_valid = (in_data == out_data) && out_header.delivered != 0;
      } else {
        is_valid = true;
      }
    }
    return is_valid;
  }

  InType GetTestInputData() final {
    return input_msg;
  }

  InType input_msg;
};

class OtcheskovSLinearTopologyMpi2ProcTests : public OtcheskovSLinearTopologyFuncTests {
 protected:
  void SetUp() override {
    if (!ppc::util::IsUnderMpirun()) {
      std::cerr << "kMPI tests are not under mpirun\n";
      GTEST_SKIP();
    }
    OtcheskovSLinearTopologyFuncTests::SetUp();
  }
};

class OtcheskovSLinearTopologyMpi4ProcTests : public OtcheskovSLinearTopologyFuncTests {
 protected:
  void SetUp() override {
    if (!ppc::util::IsUnderMpirun()) {
      std::cerr << "kMPI tests are not under mpirun\n";
      GTEST_SKIP();
    }
    OtcheskovSLinearTopologyFuncTests::SetUp();
  }
};

class OtcheskovSLinearTopologyFuncTestsValidation : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return FormatNumber(std::get<0>(test_param).first.src) + "_" + FormatNumber(std::get<0>(test_param).first.dest) +
           "_process_" + "test_" + std::to_string(std::get<1>(test_param));
  }

 protected:
  bool CheckTestOutputData(OutType &output_msg) final {
    return output_msg.first.delivered == 0;
  }

  InType GetTestInputData() final {
    return input_msg_;
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
    if (value >= 0) {
      return std::to_string(value);
    }
    return "minus_" + std::to_string(-value);
  }

  InType input_msg_;
  ppc::task::TaskPtr<InType, OutType> task_;
};

namespace {

TEST_P(OtcheskovSLinearTopologyMpi2ProcTests, Mpi2ProcsTests) {
  int proc_nums{};
  MPI_Comm_size(MPI_COMM_WORLD, &proc_nums);
  if (proc_nums < 2) {
    std::cerr << "Tests should run on 2 or more processes\n";
  } else {
    ExecuteTest(GetParam());
  }
}

TEST_P(OtcheskovSLinearTopologyMpi4ProcTests, Mpi4ProcsTests) {
  int proc_nums{};
  MPI_Comm_size(MPI_COMM_WORLD, &proc_nums);
  if (proc_nums < 4) {
    std::cerr << "Tests should run on 4 or more processes\n";
  } else {
    ExecuteTest(GetParam());
  }
}

TEST_P(OtcheskovSLinearTopologyFuncTests, SeqTests) {
  ExecuteTest(GetParam());
}

const MessageData kData5 = {1, 2, 3, 4, 5};
const MessageData kData1 = {1};

const std::array<TestType, 4> kMpiParam2Proc = {
    {{{MessageHeader{.delivered = 0, .src = 0, .dest = 0, .data_size = 5}, kData5}, 1},
     {{MessageHeader{.delivered = 0, .src = 0, .dest = 1, .data_size = 5}, kData5}, 2},
     {{MessageHeader{.delivered = 0, .src = 1, .dest = 0, .data_size = 5}, kData5}, 3},
     {{MessageHeader{.delivered = 0, .src = 1, .dest = 1, .data_size = 5}, kData5}, 4}}};

const std::array<TestType, 6> kMpiParam4Proc = {
    {{{MessageHeader{.delivered = 0, .src = 0, .dest = 2, .data_size = 5}, kData5}, 1},
     {{MessageHeader{.delivered = 0, .src = 0, .dest = 3, .data_size = 5}, kData5}, 2},
     {{MessageHeader{.delivered = 0, .src = 1, .dest = 3, .data_size = 5}, kData5}, 3},
     {{MessageHeader{.delivered = 0, .src = 3, .dest = 0, .data_size = 5}, kData5}, 4},
     {{MessageHeader{.delivered = 0, .src = 2, .dest = 1, .data_size = 5}, kData5}, 5},
     {{MessageHeader{.delivered = 0, .src = 1, .dest = 2, .data_size = 5}, kData5}, 6}}};

const std::array<TestType, 2> kSeqParam = {
    {{{MessageHeader{.delivered = 0, .src = 0, .dest = 0, .data_size = 5}, kData5}, 1},
     {{MessageHeader{.delivered = 0, .src = 0, .dest = 0, .data_size = 1}, kData1}, 2}}};

const auto kMpiTasksList2Proc = std::tuple_cat(ppc::util::AddFuncTask<OtcheskovSLinearTopologyMPI, InType>(
    kMpiParam2Proc, PPC_SETTINGS_otcheskov_s_linear_topology));

const auto kMpiTasksList4Proc = std::tuple_cat(ppc::util::AddFuncTask<OtcheskovSLinearTopologyMPI, InType>(
    kMpiParam4Proc, PPC_SETTINGS_otcheskov_s_linear_topology));

const auto kSeqTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<OtcheskovSLinearTopologySEQ, InType>(kSeqParam, PPC_SETTINGS_otcheskov_s_linear_topology));

const auto kMpiGtestValues2Proc = ppc::util::ExpandToValues(kMpiTasksList2Proc);
const auto kMpiGtestValues4Proc = ppc::util::ExpandToValues(kMpiTasksList4Proc);
const auto kSeqGtestValues = ppc::util::ExpandToValues(kSeqTasksList);

const auto kFuncTestName = OtcheskovSLinearTopologyFuncTests::PrintFuncTestName<OtcheskovSLinearTopologyFuncTests>;

INSTANTIATE_TEST_SUITE_P(Mpi2ProcsTests, OtcheskovSLinearTopologyMpi2ProcTests, kMpiGtestValues2Proc, kFuncTestName);
INSTANTIATE_TEST_SUITE_P(Mpi4ProcsTests, OtcheskovSLinearTopologyMpi4ProcTests, kMpiGtestValues4Proc, kFuncTestName);
INSTANTIATE_TEST_SUITE_P(SeqTests, OtcheskovSLinearTopologyFuncTests, kSeqGtestValues, kFuncTestName);

TEST_P(OtcheskovSLinearTopologyFuncTestsValidation, Validation) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kValidationTestParam = {{
    {{MessageHeader{.delivered = 0, .src = -1, .dest = 0, .data_size = 1}, kData1}, 1},
    {{MessageHeader{.delivered = 0, .src = 0, .dest = -1, .data_size = 1}, kData1}, 2},
    {{MessageHeader{.delivered = 0, .src = 0, .dest = 0, .data_size = 0}, MessageData{}}, 3},
    {{MessageHeader{.delivered = 1, .src = 0, .dest = 0, .data_size = 1}, kData1}, 4},
    {{MessageHeader{.delivered = 0, .src = 100000, .dest = 0, .data_size = 1}, kData1}, 5},
    {{MessageHeader{.delivered = 0, .src = 0, .dest = 100000, .data_size = 1}, kData1}, 6},
}};

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
