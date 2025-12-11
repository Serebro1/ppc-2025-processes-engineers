#include <gtest/gtest.h>

#include <cstdint>

#include "kolotukhin_a_elem_vec_sum/common/include/common.hpp"
#include "kolotukhin_a_elem_vec_sum/mpi/include/ops_mpi.hpp"
#include "kolotukhin_a_elem_vec_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kolotukhin_a_elem_vec_sum {

class KolotukhinAElemVecSumPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  std::uint64_t kCount_ = 100000000;
  std::int64_t test_answer_ = 500150000000;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == test_answer_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KolotukhinAElemVecSumPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KolotukhinAElemVecSumMPI, KolotukhinAElemVecSumSEQ>(
    PPC_SETTINGS_kolotukhin_a_elem_vec_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KolotukhinAElemVecSumPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KolotukhinAElemVecSumPerfTest, kGtestValues, kPerfTestName);

}  // namespace kolotukhin_a_elem_vec_sum
