#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "tsyplakov_k_vec_neighbours/common/include/common.hpp"
#include "tsyplakov_k_vec_neighbours/mpi/include/ops_mpi.hpp"
#include "tsyplakov_k_vec_neighbours/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace tsyplakov_k_vec_neighbours {

class TsyplakovKVecNeighboursPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  static const int kCount = 150000000;
  InType input_data;

  void SetUp() override {
    std::vector<int> vec(kCount);
    for (int i = 0; i < kCount; ++i) {
      vec[i] = i;
    }
    input_data = vec;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    auto [i1, i2] = output_data;

    if (i2 != i1 + 1) {
      return false;
    }
    if (i1 < 0 || std::cmp_greater_equal(i2, input_data.size())) {
      return false;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(TsyplakovKVecNeighboursPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, TsyplakovKVecNeighboursMPI, TsyplakovKVecNeighboursSEQ>(
    PPC_SETTINGS_tsyplakov_k_vec_neighbours);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TsyplakovKVecNeighboursPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TsyplakovKVecNeighboursPerfTest, kGtestValues, kPerfTestName);

}  // namespace tsyplakov_k_vec_neighbours
