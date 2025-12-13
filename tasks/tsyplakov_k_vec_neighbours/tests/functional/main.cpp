#include <gtest/gtest.h>

#include <array>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tsyplakov_k_vec_neighbours/common/include/common.hpp"
#include "tsyplakov_k_vec_neighbours/mpi/include/ops_mpi.hpp"
#include "tsyplakov_k_vec_neighbours/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace tsyplakov_k_vec_neighbours {

class TsyplakovKVecNeighboursFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &vector = input_data_;

    if (vector.size() <= 1) {
      auto [i1, i2] = output_data;
      return (i1 == -1 && i2 == -1);
    }

    int need_index = -1;
    int need_index_plus_one = -1;
    int abs_sub = INT_MAX;

    for (int i = 0; std::cmp_less(i, vector.size() - 1); ++i) {
      int temp = std::abs(vector[i + 1] - vector[i]);
      if (temp < abs_sub) {
        need_index = i;
        need_index_plus_one = i + 1;
        abs_sub = temp;
      }
    }

    auto [i1, i2] = output_data;

    return std::cmp_equal(i1, need_index) && std::cmp_equal(i2, need_index_plus_one);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(TsyplakovKVecNeighboursFuncTests, VecMinNeighbourDiff) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 22> kTestParam = {
    std::make_tuple(std::vector<int>{0, 0, 0, 0, 0}, "nechet_zero"),
    std::make_tuple(std::vector<int>{13, 13, 13, 13, 13}, "nechet_same"),
    std::make_tuple(std::vector<int>{1, 22, -3, 4, 175}, "nechet_basic"),
    std::make_tuple(std::vector<int>{-1, -2, -3, -4, -3}, "nechet_negative"),
    std::make_tuple(std::vector<int>{-87, -87, -87, -87, -87}, "nechet_same_negative"),
    std::make_tuple(std::vector<int>{-87, -87, -87, -87, -7}, "nechet_negative_2"),
    std::make_tuple(std::vector<int>{23, 46, 78, 90, 89, 45, 134, 777}, "diff_between"),
    std::make_tuple(std::vector<int>{23, 46, 78, 90, -89, 134, 135}, "nechet_diff_end"),
    std::make_tuple(std::vector<int>{45, 46, 78, 89, 45, 134, 777}, "nechet_diff_start"),
    std::make_tuple(std::vector<int>{0, 0, 0, 0, 0, 0, 0, 0}, "chet_zero"),
    std::make_tuple(std::vector<int>{4, 4, 4, 4, 4, 4, 4, 4}, "chet_same"),
    std::make_tuple(std::vector<int>{-87, -87, -87, -87, -87, -87}, "chet_same_negative"),
    std::make_tuple(std::vector<int>{34, 43}, "small"),
    std::make_tuple(std::vector<int>{12000}, "once"),
    std::make_tuple(std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, "chet_same_diff"),
    std::make_tuple(std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9}, "nechet_same_diff"),
    std::make_tuple(std::vector<int>{1461876, 124567, 637126, 1287637, 233123, 123213, -4267821, -2372163, 3127356},
                    "alot_big_values"),
    std::make_tuple(
        std::vector<int>{10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000},
        "step_values"),
    std::make_tuple(std::vector<int>{900, 900, 0, 900, -900, 9900, -9000, 900900900, -900900900, 0, 12345430, -111, 900,
                                     555, 900, 900},
                    "from_k_tsyplakov"),
    std::make_tuple(
        std::vector<int>{900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900},
        "sber"),
    std::make_tuple(std::vector<int>{21, -21, 21, -21, 21, 21, -21, -21, 3}, "school_21"),
    std::make_tuple(std::vector<int>{}, "empty")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<TsyplakovKVecNeighboursMPI, InType>(kTestParam, PPC_SETTINGS_tsyplakov_k_vec_neighbours),
    ppc::util::AddFuncTask<TsyplakovKVecNeighboursSEQ, InType>(kTestParam, PPC_SETTINGS_tsyplakov_k_vec_neighbours));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TsyplakovKVecNeighboursFuncTests::PrintFuncTestName<TsyplakovKVecNeighboursFuncTests>;

INSTANTIATE_TEST_SUITE_P(VectorFuncTests, TsyplakovKVecNeighboursFuncTests, kGtestValues, kPerfTestName);

}  // namespace
}  // namespace tsyplakov_k_vec_neighbours
