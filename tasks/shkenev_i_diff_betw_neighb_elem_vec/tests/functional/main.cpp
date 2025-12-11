#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "shkenev_i_diff_betw_neighb_elem_vec/common/include/common.hpp"
#include "shkenev_i_diff_betw_neighb_elem_vec/mpi/include/ops_mpi.hpp"
#include "shkenev_i_diff_betw_neighb_elem_vec/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shkenev_i_diff_betw_neighb_elem_vec {

class ShkenevIDiffBetwNeighbElemVecFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &vec = std::get<0>(test_param);
    int expected = std::get<1>(test_param);
    return std::to_string(vec.size()) + "_expected_" + std::to_string(expected);
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_{};
};

namespace {

const std::array<TestType, 10> kTestParam = {std::make_tuple(std::vector<int>{1, 5}, 4),
                                             std::make_tuple(std::vector<int>{8, 10, 2, 4}, 8),
                                             std::make_tuple(std::vector<int>{1, 2, 3, 4, 5}, 1),
                                             std::make_tuple(std::vector<int>{100, 5, 8, 3, 4}, 95),
                                             std::make_tuple(std::vector<int>{-1, -5, 10, 8, -7}, 15),
                                             std::make_tuple(std::vector<int>{1, 1, 1, 1, 1}, 0),
                                             std::make_tuple(std::vector<int>{0, 0, 5, 0, 0, 0}, 5),
                                             std::make_tuple(std::vector<int>{100, 50, 75, 25}, 50),
                                             std::make_tuple(std::vector<int>{5}, 0),
                                             std::make_tuple(std::vector<int>{}, 0)};

TEST_P(ShkenevIDiffBetwNeighbElemVecFuncTests, FindMaxDiff) {
  ExecuteTest(GetParam());
}

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<ShkenevIDiffBetwNeighbElemVecMPI, InType>(
                                               kTestParam, PPC_SETTINGS_shkenev_i_diff_betw_neighb_elem_vec),
                                           ppc::util::AddFuncTask<ShkenevIDiffBetwNeighbElemVecSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_shkenev_i_diff_betw_neighb_elem_vec));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    ShkenevIDiffBetwNeighbElemVecFuncTests::PrintFuncTestName<ShkenevIDiffBetwNeighbElemVecFuncTests>;

INSTANTIATE_TEST_SUITE_P(FindMaxDiff, ShkenevIDiffBetwNeighbElemVecFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace shkenev_i_diff_betw_neighb_elem_vec
