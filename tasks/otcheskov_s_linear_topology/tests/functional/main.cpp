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
           "_process";
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_data_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    bool is_valid = output_data.delivered == expected_data_.delivered && output_data.hops == expected_data_.hops &&
                    compareData(output_data.path, expected_data_.path) &&
                    compareData(output_data.received_data, expected_data_.received_data);
    return is_valid;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  OutType expected_data_{};

  bool compareData(std::vector<int> actual_data, std::vector<int> expected_data) {
    return actual_data.size() == expected_data.size() &&
           std::equal(actual_data.begin(), actual_data.end(), expected_data.begin());
  }
};

namespace {

TEST_P(OtcheskovSLinearTopologyFuncTests, LinearTopologyFuncTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 7> kTestParam = {
    std::make_tuple(Message{1, 1, {1, 2, 3, 4, 5}}, Response{true, 0, {1}, {1, 2, 3, 4, 5}, 1, 1}),
    std::make_tuple(Message{1, 2, {1, 2, 3, 4, 5}}, Response{true, 1, {1, 2}, {1, 2, 3, 4, 5}, 1, 2}),
    std::make_tuple(Message{1, 3, {1, 2, 3, 4, 5}}, Response{true, 2, {1, 2, 3}, {1, 2, 3, 4, 5}, 1, 3}),
    std::make_tuple(Message{1, 4, {1, 2, 3, 4, 5}}, Response{true, 3, {1, 2, 3, 4}, {1, 2, 3, 4, 5}, 1, 4}),
    std::make_tuple(Message{4, 1, {1, 2, 3, 4, 5}}, Response{true, 3, {4, 3, 2, 1}, {1, 2, 3, 4, 5}, 4, 1}),
    std::make_tuple(Message{3, 2, {1, 2, 3, 4, 5}}, Response{true, 1, {3, 2}, {1, 2, 3, 4, 5}, 3, 2}),
    std::make_tuple(Message{2, 3, {1, 2, 3, 4, 5}}, Response{true, 1, {2, 3}, {1, 2, 3, 4, 5}, 2, 3})};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<OtcheskovSLinearTopologyMPI, InType>(kTestParam, PPC_SETTINGS_otcheskov_s_linear_topology),
    ppc::util::AddFuncTask<OtcheskovSLinearTopologySEQ, InType>(kTestParam, PPC_SETTINGS_otcheskov_s_linear_topology));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = OtcheskovSLinearTopologyFuncTests::PrintFuncTestName<OtcheskovSLinearTopologyFuncTests>;
INSTANTIATE_TEST_SUITE_P(LinearTopologyFuncTests, OtcheskovSLinearTopologyFuncTests, kGtestValues, kFuncTestName);

}  // namespace

}  // namespace otcheskov_s_linear_topology
