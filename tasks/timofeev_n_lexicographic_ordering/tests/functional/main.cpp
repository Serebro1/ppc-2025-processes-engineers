#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>

#include "timofeev_n_lexicographic_ordering/common/include/common.hpp"
#include "timofeev_n_lexicographic_ordering/mpi/include/ops_mpi.hpp"
#include "timofeev_n_lexicographic_ordering/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace timofeev_n_lexicographic_ordering {

class TimofeevNRunFuncTestsLexicographic : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param).first + "_" + std::get<1>(test_param).second + "_" + std::get<2>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<1>(params);
    expected_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.first == expected_data_.first && output_data.second == expected_data_.second;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = std::make_pair("", "");
  OutType expected_data_;
};

namespace {

TEST_P(TimofeevNRunFuncTestsLexicographic, StringSortedTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 7> kTestParam = {
    std::make_tuple(std::make_pair(1, 1), std::make_pair("abcd", "abcd"), "sorted1sorted1"),
    std::make_tuple(std::make_pair(1, 0), std::make_pair("abcd", "abdc"), "sorted1unsorted0"),
    std::make_tuple(std::make_pair(0, 1), std::make_pair("abdc", "abcd"), "unsorted0sorted1"),
    std::make_tuple(std::make_pair(0, 0), std::make_pair("abdc", "abdc"), "unsorted0unsorted0"),
    std::make_tuple(std::make_pair(1, 1), std::make_pair("", "abcd"), "empty1sorted1"),
    std::make_tuple(std::make_pair(0, 1), std::make_pair("abdc", ""), "unsorted0empty1"),
    std::make_tuple(std::make_pair(1, 1), std::make_pair("", ""), "empty1empty1")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<TimofeevNLexicographicOrderingMPI, InType>(
                                               kTestParam, PPC_SETTINGS_timofeev_n_lexicographic_ordering),
                                           ppc::util::AddFuncTask<TimofeevNLexicographicOrderingSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_timofeev_n_lexicographic_ordering));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TimofeevNRunFuncTestsLexicographic::PrintFuncTestName<TimofeevNRunFuncTestsLexicographic>;

INSTANTIATE_TEST_SUITE_P(StringSortedTests, TimofeevNRunFuncTestsLexicographic, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace timofeev_n_lexicographic_ordering
