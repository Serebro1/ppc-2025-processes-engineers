#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include "korolev_k_string_word_count/common/include/common.hpp"
#include "korolev_k_string_word_count/mpi/include/ops_mpi.hpp"
#include "korolev_k_string_word_count/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace korolev_k_string_word_count_processes {

using korolev_k_string_word_count::InType;
using korolev_k_string_word_count::OutType;

class KorolevKRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr int kRepeat = 300000;

 protected:
  void SetUp() override {
    std::string unit = "word ";
    input_data_.reserve(unit.size() * static_cast<std::size_t>(kRepeat));
    for (int i = 0; i < kRepeat; ++i) {
      input_data_ += unit;
    }
    expected_ = kRepeat;
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

TEST_P(KorolevKRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, korolev_k_string_word_count::KorolevKStringWordCountMPI,
                                                       korolev_k_string_word_count::KorolevKStringWordCountSEQ>(
    PPC_SETTINGS_korolev_k_string_word_count);

const auto kPerfGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = KorolevKRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(PerfStringWordCount, KorolevKRunPerfTestProcesses, kPerfGtestValues, kPerfTestName);

}  // namespace korolev_k_string_word_count_processes
