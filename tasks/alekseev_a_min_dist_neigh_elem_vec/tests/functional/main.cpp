#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "alekseev_a_min_dist_neigh_elem_vec/common/include/common.hpp"
#include "alekseev_a_min_dist_neigh_elem_vec/mpi/include/ops_mpi.hpp"
#include "alekseev_a_min_dist_neigh_elem_vec/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace alekseev_a_min_dist_neigh_elem_vec {

class AlekseevAMinDistNeighElemVecRunFuncTestsProcesses
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
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
    const auto &vec = input_data_;

    int total_size = static_cast<int>(vec.size());
    if (total_size < 2) {
      return true;
    }

    int index = -1;
    int index_value = std::numeric_limits<int>::max();
    for (int i = 0; i < (static_cast<int>(vec.size()) - 1); i++) {
      int value = std::abs(vec[i + 1] - vec[i]);
      if (value < index_value) {
        index_value = value;
        index = i;
      }
    }

    auto [first, second] = output_data;
    return std::cmp_equal(first, index) && std::cmp_equal(second, index + 1);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(AlekseevAMinDistNeighElemVecRunFuncTestsProcesses, MinNeighDiffTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 20> kTestParam = {
    std::make_tuple(std::vector<int>{1, 2, 3, 4}, "basic_test"),
    std::make_tuple(std::vector<int>{0, 0, 0, 0}, "all_zero"),
    std::make_tuple(std::vector<int>{5, 5, 5, 5}, "all_same_positive"),
    std::make_tuple(std::vector<int>{-2, -2, -2, -2}, "all_same_negative"),
    std::make_tuple(std::vector<int>{-1, -2, -3, -4}, "all_negative_descending"),
    std::make_tuple(std::vector<int>{-4, -3, -2, -1}, "all_negative_ascending"),
    std::make_tuple(std::vector<int>{-2, 3, -1, 4}, "mixed_positive_negative"),
    std::make_tuple(std::vector<int>{1, 3, 2, 4}, "min_diff_at_beginning"),
    std::make_tuple(std::vector<int>{5, 8, 1, 2}, "min_diff_at_end"),
    std::make_tuple(std::vector<int>{10, 5, 7, 6}, "min_diff_in_middle"),
    std::make_tuple(std::vector<int>{1, 2}, "minimum_size_two_elements"),
    std::make_tuple(std::vector<int>{100}, "single_element_special_case"),
    std::make_tuple(std::vector<int>{}, "empty_array_special_case"),
    std::make_tuple(std::vector<int>{1, 100, 2, 101}, "one_large_diff_one_small"),
    std::make_tuple(std::vector<int>{0, 1, -1, 2}, "around_zero_values"),
    std::make_tuple(std::vector<int>{1, 4, 2, 5, 3}, "alternating_pattern"),
    std::make_tuple(std::vector<int>{1, 100, 102}, "3_elem"),
    std::make_tuple(std::vector<int>{4231, 1876, 9542, 6321, 2897, 7456, 1123, 5987, 8642, 3254, 7012,
                                     4589, 2398, 8765, 5432, 1678, 7890, 3456, 9123, 6789, 2345, 8210},
                    "large_numbers_varied"),
    std::make_tuple(std::vector<int>{1000, 9000, 2000, 8000, 3000, 7000, 4000, 6000, 5000, 8500},
                    "alternating_jumps_large"),
    std::make_tuple(std::vector<int>{9999, 1, 8888, 123, 7777, 456, 6666, 789, 5555, 10000}, "boundary_values_large")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<AlekseevAMinDistNeighElemVecMPI, InType>(
                                               kTestParam, PPC_SETTINGS_alekseev_a_min_dist_neigh_elem_vec),
                                           ppc::util::AddFuncTask<AlekseevAMinDistNeighElemVecSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_alekseev_a_min_dist_neigh_elem_vec));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = AlekseevAMinDistNeighElemVecRunFuncTestsProcesses::PrintFuncTestName<
    AlekseevAMinDistNeighElemVecRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MinNeighDiffTests, AlekseevAMinDistNeighElemVecRunFuncTestsProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace alekseev_a_min_dist_neigh_elem_vec
