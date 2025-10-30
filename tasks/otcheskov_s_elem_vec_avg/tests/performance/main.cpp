#include <gtest/gtest.h>

#include "otcheskov_s_elem_vec_avg/common/include/common.hpp"
#include "otcheskov_s_elem_vec_avg/mpi/include/ops_mpi.hpp"
#include "otcheskov_s_elem_vec_avg/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace otcheskov_s_elem_vec_avg {

class OtcheskovSElemVecAvgPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kVectorSize_ = 1000;
  InType input_data_{};

  void SetUp() override {
    input_data_.resize(kVectorSize_);
    for (int i = 0; i < kVectorSize_; ++i) {
      input_data_[i] = i + 1;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int expected_avg = (kVectorSize_ + 1) / 2;
    return expected_avg == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(OtcheskovSElemVecAvgPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, OtcheskovSElemVecAvgMPI, OtcheskovSElemVecAvgSEQ>(PPC_SETTINGS_otcheskov_s_elem_vec_avg);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = OtcheskovSElemVecAvgPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, OtcheskovSElemVecAvgPerfTests, kGtestValues, kPerfTestName);

}  // namespace otcheskov_s_elem_vec_avg
