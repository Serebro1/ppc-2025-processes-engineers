#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <random>

#include "makoveeva_s_number_of_sentence/common/include/common.hpp"
#include "makoveeva_s_number_of_sentence/mpi/include/ops_mpi.hpp"
#include "makoveeva_s_number_of_sentence/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace makoveeva_s_number_of_sentence {

class MakoveevaSNumberOfSentencePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_output_ = 0;

  void SetUp() override {
    const int num_sentences = 1000000;
    input_data_.clear();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> word_count_dist(3, 10);
    std::uniform_int_distribution<> word_len_dist(3, 8);
    std::uniform_int_distribution<> punct_dist(0, 2);

    expected_output_ = 0;

    for (int i = 0; i < num_sentences; ++i) {
      int word_count = word_count_dist(gen);
      for (int j = 0; j < word_count; ++j) {
        int word_len = word_len_dist(gen);
        for (int k = 0; k < word_len; ++k) {
          input_data_ += static_cast<char>('a' + (gen() % 26));
        }
        if (j < word_count - 1) {
          input_data_ += ' ';
        }
      }

      constexpr std::array<char, 3> kPunctuation = {'.', '!', '?'};
      input_data_ += kPunctuation.at(static_cast<size_t>(punct_dist(gen)));
      input_data_ += ' ';
      expected_output_++;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_output_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(MakoveevaSNumberOfSentencePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

#ifndef PPC_SETTINGS_makoveeva_s_number_of_sentence
#  define PPC_SETTINGS_makoveeva_s_number_of_sentence "makoveeva_s_number_of_sentence"
#endif

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SentencesCounterMPI, SentencesCounterSEQ>(
    PPC_SETTINGS_makoveeva_s_number_of_sentence);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MakoveevaSNumberOfSentencePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MakoveevaSNumberOfSentencePerfTests, kGtestValues, kPerfTestName);

}  // namespace makoveeva_s_number_of_sentence
