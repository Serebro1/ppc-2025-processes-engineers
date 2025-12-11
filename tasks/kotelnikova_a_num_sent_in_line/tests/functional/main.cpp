#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>

#include "kotelnikova_a_num_sent_in_line/common/include/common.hpp"
#include "kotelnikova_a_num_sent_in_line/mpi/include/ops_mpi.hpp"
#include "kotelnikova_a_num_sent_in_line/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kotelnikova_a_num_sent_in_line {

class KotelnikovaARunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    static int test_counter = 0;
    test_counter++;

    const std::string &text = std::get<0>(test_param);
    std::size_t expected = std::get<1>(test_param);

    std::string name = "test_" + std::to_string(test_counter);
    name += "_len" + std::to_string(text.length());
    name += "_exp" + std::to_string(expected);
    return name;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_count_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_count_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  std::size_t expected_count_ = 0;
};

namespace {

TEST_P(KotelnikovaARunFuncTestsProcesses, SentenceCountingTests) {
  ExecuteTest(GetParam());
}

struct TestFile {
  std::string filename;
  std::size_t expected;
};

std::array<TestType, 6> LoadTestData() {
  std::array<TestType, 6> test_cases;

  std::array<TestFile, 6> test_files = {{{.filename = "test_1.txt", .expected = 1},
                                         {.filename = "test_2.txt", .expected = 3},
                                         {.filename = "test_3.txt", .expected = 8},
                                         {.filename = "test_4.txt", .expected = 11},
                                         {.filename = "test_5.txt", .expected = 1},
                                         {.filename = "test_6.txt", .expected = 1312}}};

  for (size_t i = 0; i < test_files.size(); ++i) {
    const auto &file_info = test_files.at(i);

    std::string path = "tasks/kotelnikova_a_num_sent_in_line/data/" + file_info.filename;

    std::ifstream file(path);
    if (file.is_open()) {
      std::stringstream ss;
      ss << file.rdbuf();
      std::string content = ss.str();
      test_cases.at(i) = std::make_tuple(content, file_info.expected);
    }
  }

  return test_cases;
}

const std::array<TestType, 6> kTestParam = LoadTestData();

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<KotelnikovaANumSentInLineMPI, InType>(
                                               kTestParam, PPC_SETTINGS_kotelnikova_a_num_sent_in_line),
                                           ppc::util::AddFuncTask<KotelnikovaANumSentInLineSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_kotelnikova_a_num_sent_in_line));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KotelnikovaARunFuncTestsProcesses::PrintFuncTestName<KotelnikovaARunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(SentenceCountingTests, KotelnikovaARunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kotelnikova_a_num_sent_in_line
