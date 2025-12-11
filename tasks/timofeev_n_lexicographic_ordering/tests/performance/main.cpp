#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <utility>

#include "timofeev_n_lexicographic_ordering/common/include/common.hpp"
#include "timofeev_n_lexicographic_ordering/mpi/include/ops_mpi.hpp"
#include "timofeev_n_lexicographic_ordering/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace timofeev_n_lexicographic_ordering {

class LexicographicRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  // const int kCount_ = 100;
  OutType expected_data_;
  InType input_data_;
  //
  void SetUp() override {
    std::string s_perf;
    for (size_t i = 0; i < static_cast<size_t>(5e7); i++) {
      s_perf += 'a';
    }
    input_data_ = std::make_pair(s_perf, s_perf);
    expected_data_ = std::make_pair(1, 1);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.first == expected_data_.first && output_data.second == expected_data_.second;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LexicographicRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TimofeevNLexicographicOrderingMPI, TimofeevNLexicographicOrderingSEQ>(
        PPC_SETTINGS_timofeev_n_lexicographic_ordering);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LexicographicRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LexicographicRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace timofeev_n_lexicographic_ordering
