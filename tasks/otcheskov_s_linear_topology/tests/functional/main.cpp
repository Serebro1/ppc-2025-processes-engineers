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
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return false;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

namespace {

TEST_P(OtcheskovSLinearTopologyFuncTests, ) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 0> kTestParam = {};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<OtcheskovSLinearTopologyMPI, InType>(kTestParam, PPC_SETTINGS_otcheskov_s_linear_topology),
    ppc::util::AddFuncTask<OtcheskovSLinearTopologySEQ, InType>(kTestParam, PPC_SETTINGS_otcheskov_s_linear_topology));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = OtcheskovSLinearTopologyFuncTests::PrintFuncTestName<NesterovARunFuncTestsProcesses3>;

}  // namespace

}  // namespace otcheskov_s_linear_topology
