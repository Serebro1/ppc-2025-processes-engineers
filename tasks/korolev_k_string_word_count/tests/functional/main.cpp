#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <string>
#include <tuple>

#include "korolev_k_string_word_count/common/include/common.hpp"
#include "korolev_k_string_word_count/mpi/include/ops_mpi.hpp"
#include "korolev_k_string_word_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace korolev_k_string_word_count_processes {

using korolev_k_string_word_count::InType;
using korolev_k_string_word_count::OutType;
using korolev_k_string_word_count::TestType;

class KorolevKRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string s = std::get<0>(test_param);
    int expected = std::get<1>(test_param);
    std::string name;
    name.reserve(std::min<std::size_t>(s.size(), std::size_t{20}));

    for (char c : s) {
      auto uc = static_cast<unsigned char>(c);
      if (std::isalnum(uc) != 0) {
        name.push_back(c);
      } else {
        name.push_back('_');
      }
      if (name.size() >= 20) {
        break;
      }
    }
    if (name.empty()) {
      name = "empty";
    }
    return name + "_" + std::to_string(expected);
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

const std::array<TestType, 12> kTestParam = {
    std::make_tuple(std::string{""}, 0),
    std::make_tuple(std::string{"   "}, 0),
    std::make_tuple(std::string{"hello"}, 1),
    std::make_tuple(std::string{"hello world"}, 2),
    std::make_tuple(std::string{"one  two   three"}, 3),
    std::make_tuple(std::string{"  leading and trailing  "}, 3),
    std::make_tuple(std::string{"newline\nseparated\nwords"}, 3),
    std::make_tuple(std::string{"\t tabs\tand  spaces \n mix"}, 4),
    std::make_tuple(std::string{"punctuation,shouldn't-break!words?"}, 1),
    std::make_tuple(std::string{"русский  текст  да"}, 3),
    std::make_tuple(std::string{"emoji 👍🏽rocks"}, 2),
    std::make_tuple(std::string{"multi\nline with \t mixed\t\n whitespace"}, 5)};

TEST_P(KorolevKRunFuncTestsProcesses, CountWords) {
  ExecuteTest(GetParam());
}

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<korolev_k_string_word_count::KorolevKStringWordCountMPI, InType>(
                       kTestParam, PPC_SETTINGS_korolev_k_string_word_count),
                   ppc::util::AddFuncTask<korolev_k_string_word_count::KorolevKStringWordCountSEQ, InType>(
                       kTestParam, PPC_SETTINGS_korolev_k_string_word_count));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kFuncTestName = KorolevKRunFuncTestsProcesses::PrintFuncTestName<KorolevKRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(StringWordCountTests, KorolevKRunFuncTestsProcesses, kGtestValues, kFuncTestName);

}  // namespace
}  // namespace korolev_k_string_word_count_processes
