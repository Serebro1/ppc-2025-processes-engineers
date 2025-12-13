#include <gtest/gtest.h>

#include <array>
#include <string>
#include <tuple>

#include "baranov_a_sign_alternations/common/include/common.hpp"
#include "baranov_a_sign_alternations/mpi/include/ops_mpi.hpp"
#include "baranov_a_sign_alternations/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace baranov_a_sign_alternations {

class BaranovASignAternationsFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto param = GetParam();

    int test_case = std::get<0>(std::get<2>(param));

    switch (test_case) {
      case 1:
        input_data_ = {1, -1, 1, -1};
        expected_output_ = 3;
        break;
      case 2:
        input_data_ = {1, 0, -1, 2, -2};
        expected_output_ = 2;
        break;
      case 3:
        input_data_ = {1, 2, 3, 4};
        expected_output_ = 0;
        break;
      case 4:
        input_data_ = {1, -1, 1, 1, 2, -3};
        expected_output_ = 3;
        break;
      case 5:
        input_data_ = {-1, -2, -3, -4};
        expected_output_ = 0;
        break;
      case 6:
        input_data_ = {0, 1, 0, -1, 0, 2, 0, -2, 0};
        expected_output_ = 0;
        break;
      case 7:
        input_data_ = {1, 2, 3, -4};
        expected_output_ = 1;
        break;
      case 8:
        input_data_ = {};
        expected_output_ = 0;
        break;
      case 9:
        input_data_ = {1, -1};
        expected_output_ = 1;
        break;
      case 10:
        input_data_ = {1, 2};
        expected_output_ = 0;
        break;
      case 11:
        input_data_ = {0};
        expected_output_ = 0;
        break;
      default:
        input_data_ = {1, -1, 1, -1};
        expected_output_ = 3;
        return;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_output_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_{};
};

namespace {

TEST_P(BaranovASignAternationsFuncTests, SignAlternationsTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 11> kTestParam = {
    std::make_tuple(1, "simple_alternations"),
    std::make_tuple(2, "with_zero"),
    std::make_tuple(3, "all_positive"),
    std::make_tuple(4, "hard_vector"),
    std::make_tuple(5, "all_negative"),
    std::make_tuple(6, "multiple_zeros"),
    std::make_tuple(7, "single_alternation"),
    std::make_tuple(8, "empty_vector"),
    std::make_tuple(9, "two_elements_with_alternation"),
    std::make_tuple(10, "two_elements_no_alternation"),
    std::make_tuple(11, "only_one_zero"),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<BaranovASignAlternationsMPI, InType>(kTestParam, PPC_SETTINGS_baranov_a_sign_alternations),
    ppc::util::AddFuncTask<BaranovASignAlternationsSEQ, InType>(kTestParam, PPC_SETTINGS_baranov_a_sign_alternations));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = BaranovASignAternationsFuncTests::PrintFuncTestName<BaranovASignAternationsFuncTests>;

INSTANTIATE_TEST_SUITE_P(SignAlternationsTests, BaranovASignAternationsFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace baranov_a_sign_alternations
