#include <gtest/gtest.h>

#include <array>
#include <cctype>
#include <cstddef>
#include <functional>
#include <string>
#include <tuple>

#include "makoveeva_s_number_of_sentence/common/include/common.hpp"
#include "makoveeva_s_number_of_sentence/mpi/include/ops_mpi.hpp"
#include "makoveeva_s_number_of_sentence/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace makoveeva_s_number_of_sentence {

class MakoveevaSNumberOfSentenceRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string input = std::get<0>(test_param);
    std::string expected = std::get<1>(test_param);

    std::string sanitized;
    sanitized.reserve(input.length() + expected.length() + 20);

    for (char ch : input) {
      if (std::isalnum(static_cast<unsigned char>(ch)) != 0) {
        sanitized += ch;
      } else {
        sanitized += '_';
      }
    }

    std::size_t input_hash = std::hash<std::string>{}(input);
    sanitized += "_count_";
    sanitized += expected;
    sanitized += "_";
    sanitized += std::to_string(input_hash % 10000);

    return sanitized;
  }

 protected:
  void SetUp() override {
    auto params = std::get<2>(GetParam());
    input_data_ = std::get<0>(params);
    expected_output_ = std::stoi(std::get<1>(params));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_output_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_ = 0;
};

namespace {

TEST_P(MakoveevaSNumberOfSentenceRunFuncTestsProcesses, CountSentences) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 27> kTestParam = {
    std::make_tuple("Hello world.", "1"),
    std::make_tuple("Hello! How are you?", "2"),
    std::make_tuple("This is a test. Another sentence! And one more?", "3"),
    std::make_tuple("", "0"),
    std::make_tuple("Na sentences here", "0"),
    std::make_tuple("One. Two. Three.", "3"),
    std::make_tuple("Multiple punctuation...!!!", "1"),
    std::make_tuple("Mix. Of! Different? Endings.", "4"),
    std::make_tuple("Only dots...", "1"),
    std::make_tuple("Single! Exclamation!", "2"),
    std::make_tuple("Question? Answer! Statement.", "3"),
    std::make_tuple("Wow!!! Amazing!!! Great!!!", "3"),
    std::make_tuple("A.B.C.D.E.F.G.H.I.J.", "10"),
    std::make_tuple("Just one very long sentence without any ending", "0"),
    std::make_tuple("Start. Middle! End?", "3"),
    std::make_tuple("A.", "1"),
    std::make_tuple(".!?", "1"),
    std::make_tuple("!?.", "1"),
    std::make_tuple("Hello . World !", "2"),
    std::make_tuple("A.B.C", "2"),
    std::make_tuple("abc..def.", "2"),
    std::make_tuple("abc...def.", "2"),
    std::make_tuple("abc!!!def.", "2"),
    std::make_tuple("abc.!?def.", "2"),
    std::make_tuple(".", "1"),
    std::make_tuple(" . ", "1"),
    std::make_tuple("a.b.c.d.e.f.g.h.i.j.l.l.m.n.o.p.q.r.s.t.u.v.w.x.y.w.", "26"),
};

#ifndef PPC_SETTINGS_makoveeva_s_number_of_sentence
#  define PPC_SETTINGS_makoveeva_s_number_of_sentence "makoveeva_s_number_of_sentence"
#endif

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<SentencesCounterMPI, InType>(kTestParam, PPC_SETTINGS_makoveeva_s_number_of_sentence),
    ppc::util::AddFuncTask<SentencesCounterSEQ, InType>(kTestParam, PPC_SETTINGS_makoveeva_s_number_of_sentence));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    MakoveevaSNumberOfSentenceRunFuncTestsProcesses::PrintFuncTestName<MakoveevaSNumberOfSentenceRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(SentenceCountingTests, MakoveevaSNumberOfSentenceRunFuncTestsProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace makoveeva_s_number_of_sentence
