#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "alekseev_a_min_dist_neigh_elem_vec/common/include/common.hpp"
#include "alekseev_a_min_dist_neigh_elem_vec/mpi/include/ops_mpi.hpp"
#include "alekseev_a_min_dist_neigh_elem_vec/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace alekseev_a_min_dist_neigh_elem_vec {

class AlekseevAMinDistNeighElemVecRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    const int vector_size = 100000000;
    std::vector<int> vec(vector_size);
    for (int i = 0; i < vector_size; i++) {
      vec[i] = (i % 1000 + 7) * 3;
    }
    input_data_ = vec;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    auto [first, second] = output_data;
    return (second == first + 1) && (first >= 0) && (std::cmp_less(second, input_data_.size()));
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(AlekseevAMinDistNeighElemVecRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, AlekseevAMinDistNeighElemVecMPI, AlekseevAMinDistNeighElemVecSEQ>(
        PPC_SETTINGS_alekseev_a_min_dist_neigh_elem_vec);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = AlekseevAMinDistNeighElemVecRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, AlekseevAMinDistNeighElemVecRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace alekseev_a_min_dist_neigh_elem_vec
